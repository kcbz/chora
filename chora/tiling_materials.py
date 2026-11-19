"""Source-backed tiling records and explicit simulation-only constitutive models.

Manufacturer facts anchor geometry, density (where supplied), and time windows.
Herschel–Bulkley/thixotropy and transport coefficients are declared assumptions.
This module does not infer fracture from a test-machine breaking load.
"""
from dataclasses import dataclass
from pathlib import Path
from functools import lru_cache
from copy import deepcopy
import math
import numpy as np
import yaml

from .paths import WORKSPACE_ROOT as ROOT

@lru_cache(maxsize=2)
def _catalog_at_revision(mtime_ns):
    return yaml.safe_load((ROOT/'configs/materials/tiling_baseline.yaml').read_text())

def catalog():
    file=ROOT/'configs/materials/tiling_baseline.yaml'
    return deepcopy(_catalog_at_revision(file.stat().st_mtime_ns))

def value(product,key):
    p=catalog()['products'][product]
    return (p['published'].get(key) or p['assumptions'][key])['value']

def tile_properties():
    dims=np.array(value('tile','dimensions'),float);density=value('tile','density');mass=float(np.prod(dims)*density)
    return dict(name=catalog()['products']['tile']['name'],dimensions_m=dims.tolist(),mass_kg=mass,
      mass_status='derived_from_assumed_density_not_measured',density_kg_m3=density,
      inertia_kg_m2=(mass/12*np.array([dims[1]**2+dims[2]**2,dims[0]**2+dims[2]**2,dims[0]**2+dims[1]**2])).tolist(),
      breaking_test_lower_report_n=value('tile','breaking_load_lower_report'),fracture_model=False,
      friction=value('tile','sliding_friction'),hardware_validated=False)

@dataclass
class PasteState:
    kind:str
    age_s:float=0.
    structure:float=1.
    exposure_s:float|None=None

    def __post_init__(self):
        if self.kind not in {'adhesive','grout'}:raise ValueError('Unknown paste')
        if not math.isfinite(self.age_s) or self.age_s<0:raise ValueError('Invalid paste age')
        if not 0<=self.structure<=1:raise ValueError('Invalid structure')
        p=catalog()['products'][self.kind]
        self.parameters={k:r['value'] for group in ['published','assumptions'] for k,r in p[group].items()}
    @property
    def pot_life_s(self):return self.parameters.get('pot_life',self.parameters.get('pot_life_range',[0])[0])
    @property
    def workable(self):return self.age_s<self.pot_life_s
    @property
    def placement_window_open(self):
        return self.workable and (self.exposure_s is None or self.exposure_s<self.parameters.get('open_time',float('inf')))
    def advance(self,dt,shear_rate=0.):
        if not math.isfinite(dt) or dt<0 or not math.isfinite(shear_rate):raise ValueError('Invalid time/shear')
        # Exact evolution of a first-order structural state at constant shear.
        rebuild=1/self.parameters['structure_rebuild_time'];breakdown=self.parameters['structure_breakdown_rate']*abs(shear_rate)
        rate=rebuild+breakdown;equilibrium=rebuild/rate
        self.structure=equilibrium+(self.structure-equilibrium)*math.exp(-rate*dt)
        self.age_s+=dt
        if self.exposure_s is not None:self.exposure_s+=dt
    def shear_stress_pa(self,shear_rate):
        if not math.isfinite(shear_rate):raise ValueError('Invalid shear rate')
        if shear_rate==0:return 0.
        p=self.parameters;aging=1+p['aging_multiplier']*(self.age_s/self.pot_life_s)**2
        yield_stress=p['yield_stress']*(.4+.6*self.structure)*aging
        stress=yield_stress+p['consistency']*aging*abs(shear_rate)**p['flow_index']
        return math.copysign(stress,shear_rate)
    def snapshot(self,shear_rate=10.):
        stress=self.shear_stress_pa(shear_rate)
        return dict(kind=self.kind,age_s=self.age_s,exposure_s=self.exposure_s,structure=self.structure,
          workable=self.workable,placement_window_open=self.placement_window_open,
          shear_rate_per_s=shear_rate,shear_stress_pa=stress,
          apparent_viscosity_pa_s=abs(stress/shear_rate) if shear_rate else None,
          law='assumed age-dependent Herschel-Bulkley with first-order thixotropy',hardware_validated=False)

class MaterialInventory:
    """Finite-volume ledgers for a 2x2 panel, including waste and sponge collection.

    Operations move actual volume between compartments. This is a reduced
    transport model, not a resolved free-surface fluid solver.
    """
    def __init__(self,tile_count=4):
        if tile_count!=4:raise ValueError('Baseline panel has four tiles')
        self.tile=tile_properties();self.width=self.tile['dimensions_m'][0];self.depth=self.tile['dimensions_m'][2]
        self.joint_width=value('tile','recommended_joint_width')
        self.joint_capacity=(4*self.width*self.joint_width+self.joint_width**2)*self.depth
        self.adhesive=PasteState('adhesive');self.grout=PasteState('grout')
        self.grout_mixed=False
        self.adhesive_reservoir_m3=.001;self.adhesive_under_m3=np.zeros(4);self.adhesive_squeezeout_m3=0.
        self.grout_reservoir_m3=.0005;self.grout_joints_m3=0.;self.grout_residue_m3=np.zeros(4);self.grout_collected_m3=0.
        self.adhesive_initial_m3=self.adhesive_total_m3;self.grout_initial_m3=self.grout_total_m3
        self.spread_at_s=[None]*4;self.bedded_at_s=[None]*4
    @property
    def adhesive_total_m3(self):return self.adhesive_reservoir_m3+float(self.adhesive_under_m3.sum())+self.adhesive_squeezeout_m3
    @property
    def grout_total_m3(self):return self.grout_reservoir_m3+self.grout_joints_m3+float(self.grout_residue_m3.sum())+self.grout_collected_m3
    def advance(self,dt):
        self.adhesive.advance(dt)
        if self.grout_mixed:self.grout.advance(dt)
    def mix_grout(self):
        if self.grout_mixed:raise ValueError('This grout batch is already mixed; retempering is not modeled')
        self.grout_mixed=True;self.grout=PasteState('grout')
    def spread(self,index,volume_m3):
        if index not in range(4) or not math.isfinite(volume_m3) or volume_m3<0:raise ValueError('Invalid deposit')
        if not self.adhesive.workable:raise ValueError('Adhesive batch exceeded pot-life anchor')
        amount=min(volume_m3,self.adhesive_reservoir_m3)
        self.adhesive_reservoir_m3-=amount;self.adhesive_under_m3[index]+=amount
        if amount>0 and self.spread_at_s[index] is None:self.spread_at_s[index]=self.adhesive.age_s
        return amount
    def bed(self,index,gap_m):
        if index not in range(4) or gap_m<=0 or not math.isfinite(gap_m):raise ValueError('Invalid bedding geometry')
        if self.spread_at_s[index] is None:raise ValueError('No adhesive deposited')
        if not self.adhesive.workable or self.adhesive.age_s-self.spread_at_s[index]>=value('adhesive','open_time'):raise ValueError('Adhesive placement window exceeded')
        capacity=self.width**2*gap_m;excess=max(0.,self.adhesive_under_m3[index]-capacity)
        self.adhesive_under_m3[index]-=excess;self.adhesive_squeezeout_m3+=excess
        self.bedded_at_s[index]=self.adhesive.age_s
        return dict(under_tile_m3=float(self.adhesive_under_m3[index]),squeezeout_m3=excess,
          volume_based_coverage_upper_bound=min(1.,self.adhesive_under_m3[index]/capacity),
          coverage_is_spatially_resolved=False)
    def grout_panel(self,volume_m3):
        if not math.isfinite(volume_m3) or volume_m3<0:raise ValueError('Invalid grout volume')
        if any(t is None or self.adhesive.age_s-t<value('adhesive','minimum_before_grouting') for t in self.bedded_at_s):raise ValueError('Adhesive has not reached the published minimum grouting age')
        if not self.grout_mixed:raise ValueError('Mix grout before applying it')
        if not self.grout.workable:raise ValueError('Grout batch exceeded pot-life anchor; mix a fresh batch')
        amount=min(volume_m3,self.grout_reservoir_m3);self.grout_reservoir_m3-=amount
        fill=min(amount,max(0.,self.joint_capacity-self.grout_joints_m3));self.grout_joints_m3+=fill
        self.grout_residue_m3+=(amount-fill)/4
        return dict(joint_fill_fraction=self.grout_joints_m3/self.joint_capacity,residue_m3=float(self.grout_residue_m3.sum()))

class SurfaceGrout:
    """Spatial residue coupled to an actually contacting/moving sponge.

    Thickness is represented by finite cells; collected material is accounted
    for with finite sponge capacity. A prescribed planar layer is not a bulk
    grout-flow or pore-scale absorption model.
    """
    def __init__(self,width,thickness=.0002,cells=16,age_s=1500.,capacity_m3=6e-6):
        if width<=0 or thickness<0 or capacity_m3<0:raise ValueError('Invalid surface dimensions')
        self.width=width;self.cells=cells;self.dx=width/cells;self.cell_area=self.dx**2
        c=(np.arange(cells)+.5)*self.dx-width/2;xx,yy=np.meshgrid(c,c,indexing='ij')
        self.centers=np.c_[xx.ravel(),yy.ravel()];self.volume=np.full(cells*cells,thickness*self.cell_area)
        self.initial_m3=float(self.volume.sum());self.collected_m3=0.;self.capacity_m3=capacity_m3
        self.paste=PasteState('grout',age_s=age_s);self.work_j=0.
    @property
    def thickness(self):return self.volume/self.cell_area
    @property
    def remaining_m3(self):return float(self.volume.sum())
    def contact(self,xy,half_size,normal_force_n,speed_m_s,dt):
        if normal_force_n<0 or speed_m_s<0 or dt<0:raise ValueError('Invalid sponge contact')
        self.paste.advance(dt,speed_m_s/.001 if normal_force_n>.15 else 0.)
        if self.paste.age_s>=max(self.paste.parameters['final_set_range']):return dict(removed_m3=0.,drag_n=0.)
        if normal_force_n<=.15 or speed_m_s<=1e-5 or self.collected_m3>=self.capacity_m3:return dict(removed_m3=0.,drag_n=0.)
        mask=np.all(abs(self.centers-np.asarray(xy))<=np.asarray(half_size),axis=1)&(self.volume>0)
        if not np.any(mask):return dict(removed_m3=0.,drag_n=0.)
        p=self.paste.parameters;age_factor=math.exp(-p['residue_adhesion_age_rate']*self.paste.age_s/self.paste.pot_life_s)
        fraction=1-math.exp(-p['wipe_rate']*age_factor*(normal_force_n/5)*(speed_m_s/.03)*dt)
        requested=self.volume[mask]*fraction;amount=min(float(requested.sum()),self.capacity_m3-self.collected_m3)
        if requested.sum()>0:self.volume[mask]-=requested*(amount/requested.sum())
        self.collected_m3+=amount
        # Additional paste shear reaction, capped by a declared interface limit.
        # Equal/opposite forces must be applied by the robot adapter.
        area=int(mask.sum())*self.cell_area;stress=abs(self.paste.shear_stress_pa(speed_m_s/.001))
        drag=min(stress*area,.5*normal_force_n);self.work_j+=drag*speed_m_s*dt
        return dict(removed_m3=amount,drag_n=drag)

def squeeze_force(paste,area_m2,gap_m,approach_m_s):
    """Circular-equivalent squeeze-film diagnostic; assumed continuum law.

    Not used as an invisible native support. The circular equivalent and uniform
    layer make this a material-workbench prediction, not validated tile bedding.
    """
    if area_m2<=0 or gap_m<=0 or approach_m_s<0:raise ValueError('Invalid squeeze geometry')
    if approach_m_s==0:return 0.
    radius=math.sqrt(area_m2/math.pi);gamma=approach_m_s*radius/gap_m**2
    viscosity=abs(paste.shear_stress_pa(gamma))/max(gamma,1e-9)
    return 3*math.pi*viscosity*radius**4*approach_m_s/(2*gap_m**3)

def workbench(age_s=600.,exposure_s=300.,speed_m_s=.0001,gap_m=.003):
    if not all(math.isfinite(x) for x in [age_s,exposure_s,speed_m_s,gap_m]) or not (0<=age_s<=172800 and 0<=exposure_s<=age_s and 0<=speed_m_s<=.005 and .001<=gap_m<=.012):raise ValueError('Material workbench input out of range')
    adhesive=PasteState('adhesive',age_s=age_s,exposure_s=exposure_s);grout=PasteState('grout',age_s=age_s)
    tile=tile_properties();area=tile['dimensions_m'][0]*tile['dimensions_m'][1]
    return dict(tile=tile,adhesive=adhesive.snapshot(),grout=grout.snapshot(),
      squeeze_force_n=squeeze_force(adhesive,area,gap_m,speed_m_s) if adhesive.placement_window_open else None,
      layer_volume_ml=area*gap_m*1e6,simulation_only=True,
      warning='Flow, structure recovery, fresh tack, squeeze response and transport parameters are explicit assumptions; manufacturer facts do not validate these equations.')

def reference_panel():
    """Reproducible material-accounting sequence, not a robot expert rollout."""
    m=MaterialInventory();events=[]
    for i in range(4):
        amount=m.spread(i,m.width**2*.006);m.advance(30)
        result=m.bed(i,.003)
        events.append(dict(operation='spread_and_bed',tile=i+1,adhesive_deposited_ml=amount*1e6,
          under_tile_ml=result['under_tile_m3']*1e6,squeezeout_ml=result['squeezeout_m3']*1e6,
          coverage_upper_bound=result['volume_based_coverage_upper_bound'],spatial_coverage_evaluated=False))
    m.advance(value('adhesive','minimum_before_grouting'));m.mix_grout()
    result=m.grout_panel(m.joint_capacity+4*m.width**2*.0002)
    events.append(dict(operation='wait_then_mix_and_grout',elapsed_s=m.adhesive.age_s,
      joint_capacity_ml=m.joint_capacity*1e6,fill_fraction=result['joint_fill_fraction'],face_residue_ml=result['residue_m3']*1e6))
    return dict(scope='numerical_material_sequence_no_robot_motion',events=events,
      adhesive_volume_error_ml=(m.adhesive_total_m3-m.adhesive_initial_m3)*1e6,
      grout_volume_error_ml=(m.grout_total_m3-m.grout_initial_m3)*1e6,
      joint_capacity_ml=m.joint_capacity*1e6,installation_validated=False)
