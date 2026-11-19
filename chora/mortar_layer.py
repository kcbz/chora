"""Conservative spatial mortar bed, coupled to tools and free rigid tiles.

A reduced height-field/elastoviscoplastic foundation, not 3-D fluid or FEM.
Volume redistribution, cell contact support and assumed HB squeeze/shear response
are explicit. No geometry welding, body-pose writes, or preset bedding animation.
"""
from pathlib import Path
import yaml
import numpy as np

from .paths import WORKSPACE_ROOT
PARAM=yaml.safe_load((WORKSPACE_ROOT/'configs/materials/mortar_layer.yaml').read_text())['parameters']
from .tiling_materials import PasteState, value


class MortarLayer:
    def __init__(self, targets, width=.1524, surface=.24, cells=None):
        cells=cells or PARAM['cells_per_tile_edge']
        self.targets=np.asarray(targets,float);self.width=width;self.surface=surface
        self.cells=cells;self.dx=width/cells;self.area=self.dx**2
        axis=(np.arange(cells)+.5)*self.dx-width/2
        xy=np.stack(np.meshgrid(axis,axis,indexing='ij'),-1).reshape(-1,2)
        self.xy=self.targets[:,None,:2]+xy[None,:,:]
        self.volume=np.zeros((len(targets),cells*cells))
        self.squeezed_m3=np.zeros(len(targets));self.reservoir_m3=PARAM['reservoir_volume_m3']
        self.initial_m3=self.reservoir_m3;self.pending_m3=0.
        self.paste=PasteState('adhesive');self.spread_at=np.full(len(targets),np.nan)
        self.loaded_n=np.zeros(len(targets));self.contact_mask=np.zeros_like(self.volume,dtype=bool)
        self.peak_load_n=np.zeros(len(targets));self.work_j=0.
        self.applied_by_tool_m3=0.;self.redistributed_m3=0.
        self.combed=np.zeros_like(self.volume,dtype=bool)
        self.target_bed_m=PARAM['target_bed_m']
    @property
    def height(self):return self.volume/self.area
    @property
    def balance_error_m3(self):
        return self.reservoir_m3+float(self.volume.sum()+self.squeezed_m3.sum())-self.initial_m3
    def arm_dose(self,volume_ml):
        if volume_ml==0:self.pending_m3=0.;return 'valve_closed'
        if not self.paste.workable:return 'adhesive_expired'
        if self.pending_m3>1e-10:return 'dose_already_pending'
        if volume_ml<0:return 'invalid_dose'
        if volume_ml*1e-6>self.reservoir_m3:return 'insufficient_adhesive'
        self.pending_m3=volume_ml*1e-6
        return 'dose_armed_for_tool_motion'
    def advance(self,dt):
        self.paste.advance(dt);self.loaded_n[:]=0.;self.contact_mask[:]=False
    def apply_tool(self,previous,current,dt):
        """Moving slot applicator supplies its combed swath, at substrate height.

        A dose only leaves the finite reservoir in swept cells. Notch spacing is
        deliberately resolved by the grid. No deposit when lifted or stationary.
        The chosen volumetric metering law is cross-section times travel distance.
        """
        velocity=(np.asarray(current)-previous)/dt
        if self.pending_m3<=0 or not self.paste.workable or not (0<=current[2]-self.surface<=.002):return np.zeros(3)
        if velocity[0]<=1e-5 or abs(velocity[1])>.02:return np.zeros(3)
        # Swept blade strip, including half a grid cell to avoid time-step gaps.
        mask=(abs(self.xy[:,:,0]-current[0])<=self.dx/2)&(abs(self.xy[:,:,1]-current[1])<=self.width/2)
        # Alternating 9.525 mm gaps/teeth; the as-combed ridges are 8 mm high.
        ridges=(np.arange(self.cells*self.cells)%self.cells)%2==0
        mask &= ridges[None,:]
        if not mask.any():return np.zeros(3)
        requested=PARAM['ridge_height_m']*self.area*min(1.,velocity[0]*dt/self.dx)
        weights=mask.astype(float)*requested
        total=float(weights.sum());amount=min(total,self.pending_m3,self.reservoir_m3)
        if amount<=0:return np.zeros(3)
        addition=weights*(amount/total);self.volume+=addition
        self.reservoir_m3-=amount;self.pending_m3-=amount;self.applied_by_tool_m3+=amount
        for i in range(len(self.targets)):
            if addition[i].sum()>0 and np.isnan(self.spread_at[i]):self.spread_at[i]=self.paste.age_s
        self.combed|=mask
        # Assumed HB drag acts at the actual tool; equal reaction goes to support.
        area=self.width*.008
        drag=abs(self.paste.shear_stress_pa(velocity[0]/.008))*area
        self.work_j+=drag*velocity[0]*dt
        return np.array([-drag,0.,0.])
    def tile_forces(self,index,position,rotation,linear,angular,dt):
        """Distributed forces at intersecting underside cells, then plastic flow.

        Elastic stress k*penetration supplies contact support. Stress above an
        assumed confinement-dependent HB yield/flow stress causes height relaxation.
        Displaced volume first fills low cells within the footprint; surplus goes into a
        tracked outside-footprint squeeze-out compartment. No tensile bond/cure.
        """
        if not self.volume[index].any() or np.any(abs(position[:2]-self.targets[index,:2])>self.width):
            return np.zeros(3),np.zeros(3)
        xy=self.xy[index];R=rotation
        # Underside plane through tile bottom; world vertical intersections.
        normal=R[:,2];bottom=position-normal*.00635
        if normal[2]<.9:return np.zeros(3),np.zeros(3)
        gap=bottom[2]-self.surface-((xy-bottom[:2])@normal[:2])/normal[2]
        points=np.c_[xy,np.full(len(xy),self.surface)+gap]
        local=(points-position)@R
        footprint=np.all(abs(local[:,:2])<=self.width/2,axis=1)
        height=self.height[index];penetration=np.maximum(0.,height-gap)
        mask=footprint&(height>1e-8)&(penetration>0)
        self.contact_mask[index]|=mask
        if not mask.any():return np.zeros(3),np.zeros(3)
        velocity=linear+np.cross(np.broadcast_to(angular,points.shape),points-position)
        p=self.paste.parameters
        closing=np.maximum(0.,-velocity[:,2])
        # Damping is an explicit numerical contact parameter, recorded in config.
        stress=p['normal_stiffness']*penetration+PARAM['normal_damping_pa_s_per_m']*closing
        stress=np.where(mask,stress,0.)
        vertical=stress*self.area
        force=np.c_[np.zeros(len(xy)),np.zeros(len(xy)),vertical]
        speed=np.linalg.norm(velocity[:,:2],axis=1)
        shear_rate=speed/np.maximum(gap,.0005)
        aging=1+p['aging_multiplier']*(self.paste.age_s/self.paste.pot_life_s)**2
        shear=np.where(shear_rate>1e-8,(p['yield_stress']*(.4+.6*self.paste.structure)+p['consistency']*shear_rate**p['flow_index'])*aging,0.)
        drag=np.minimum(shear*self.area,PARAM['tangential_force_to_normal_cap']*vertical)
        force[:,:2]=-velocity[:,:2]*np.divide(drag,speed,out=np.zeros_like(speed),where=speed>1e-8)[:,None]
        load=float(vertical.sum());self.loaded_n[index]+=load
        self.peak_load_n[index]=max(self.peak_load_n[index],self.loaded_n[index])
        # Reduced Bingham squeeze resistance; coefficients are simulation choices.
        gap_positive=np.maximum(gap,.0005)
        confinement=1+PARAM['confinement_factor']*(self.width/2)/gap_positive
        flow_stress=p['yield_stress']*(.4+.6*self.paste.structure)*confinement
        viscosity=PARAM['plastic_relaxation_pa_s_per_m']*confinement
        collapse=np.minimum(penetration,np.maximum(0.,stress-flow_stress)/viscosity*dt)
        displaced=np.where(mask,collapse*self.area,0.)
        amount=float(displaced.sum());self.volume[index]-=displaced
        if amount>0:
            # Resolved void space across this footprint (lumped redistribution) beneath the current tile plane.
            room=np.where(footprint,np.maximum(0.,gap-self.height[index])*self.area,0.)
            capacity=float(room.sum());inside=min(amount,capacity)
            if capacity>0:self.volume[index]+=room*(inside/capacity)
            self.squeezed_m3[index]+=amount-inside
            self.redistributed_m3+=inside
        self.work_j+=float(np.sum(drag*speed))*dt
        return force.sum(axis=0),np.cross(points-position,force).sum(axis=0)
    def couple_tile(self,position,rotation,linear,angular,dt):
        # Contact follows world geometry, never the tile's semantic target/index.
        force=np.zeros(3);torque=np.zeros(3)
        for patch in range(len(self.targets)):
            if np.any(abs(position[:2]-self.targets[patch,:2])>self.width):continue
            f,t=self.tile_forces(patch,position,rotation,linear,angular,dt)
            force+=f;torque+=t
        return force,torque
    def snapshot(self):
        return dict(model='spatial_height_field_elastoviscoplastic_foundation',
          reservoir_ml=round(self.reservoir_m3*1e6,4),pending_dose_ml=round(self.pending_m3*1e6,4),
          under_tiles_ml=(self.volume.sum(axis=1)*1e6).round(4).tolist(),
          squeezeout_ml=(self.squeezed_m3*1e6).round(4).tolist(),
          volume_balance_error_ml=round(self.balance_error_m3*1e6,10),
          deposited_ml=round(self.applied_by_tool_m3*1e6,4),
          redistributed_ml=round(self.redistributed_m3*1e6,4),age_s=round(self.paste.age_s,3),
          open_time_s=value('adhesive','open_time'),workable=self.paste.workable,
          exposed_age_s=[None if np.isnan(t) else round(self.paste.age_s-t,3) for t in self.spread_at],
          contact_area_percent=(self.contact_mask.mean(axis=1)*100).round(2).tolist(),
          current_load_n=self.loaded_n.round(3).tolist(),peak_load_n=self.peak_load_n.round(3).tolist(),
          work_j=round(self.work_j,6),hardware_validated=False,fracture_model=False)
