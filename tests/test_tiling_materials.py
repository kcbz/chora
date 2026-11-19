import math
import numpy as np
import pytest
from chora.tiling_materials import catalog,tile_properties,PasteState,MaterialInventory,SurfaceGrout,squeeze_force,workbench,reference_panel

def test_product_facts_are_not_silently_turned_into_contact_or_fracture_parameters():
    c=catalog();t=tile_properties()
    assert t['dimensions_m']==[.1524,.1524,.0127]
    assert t['mass_kg']==pytest.approx(.6489277344)
    assert 'assumed' in t['mass_status'] and not t['fracture_model']
    assert c['products']['tile']['published']['breaking_load_lower_report']['value']==pytest.approx(3558.5772922084)
    assert c['products']['tile']['published']['wet_dcof_lower']['value']!=t['friction']
    assert not c['hardware_validated']
    assert 'Net mass of one tile' in c['products']['tile']['missing']
    for product in c['products'].values():
        assert all(p['status']=='manufacturer_reported' and p['source'] in c['sources'] for p in product['published'].values())
        assert all(p['status']=='simulation_assumption' for p in product['assumptions'].values())

def test_paste_shear_thinning_recovery_aging_and_separate_windows():
    p=PasteState('adhesive',exposure_s=0)
    assert p.shear_stress_pa(100)/100<p.shear_stress_pa(1)
    p.advance(10,50);worked=p.structure;assert 0<worked<.1
    p.advance(180,0);assert p.structure>worked
    fresh=PasteState('adhesive');aged=PasteState('adhesive',age_s=3600)
    assert aged.shear_stress_pa(10)>fresh.shear_stress_pa(10)
    exposed=PasteState('adhesive',age_s=2500,exposure_s=2450)
    assert exposed.workable and not exposed.placement_window_open
    assert not PasteState('adhesive',age_s=7200).workable
    assert not PasteState('grout',age_s=2700).workable

def test_panel_conserves_material_and_enforces_cure_and_mix_sequence():
    m=MaterialInventory()
    for i in range(4):
        m.spread(i,m.width**2*.006);m.bed(i,.003)
    assert m.adhesive_squeezeout_m3>0
    assert m.adhesive_total_m3==pytest.approx(m.adhesive_initial_m3,abs=1e-14)
    with pytest.raises(ValueError,match='minimum grouting age'):m.grout_panel(.001)
    m.advance(86401)
    with pytest.raises(ValueError,match='Mix grout'):m.grout_panel(.001)
    m.mix_grout();r=m.grout_panel(.001)
    assert r['joint_fill_fraction']==1 and m.grout_reservoir_m3==0
    assert m.grout_total_m3==pytest.approx(m.grout_initial_m3,abs=1e-14)
    assert m.grout_panel(.001)['residue_m3']==pytest.approx(r['residue_m3'])
    with pytest.raises(ValueError,match='already mixed'):m.mix_grout()
    s=reference_panel();assert s['scope']=='numerical_material_sequence_no_robot_motion'
    assert abs(s['adhesive_volume_error_ml'])<1e-9 and abs(s['grout_volume_error_ml'])<1e-9

def test_expired_or_absent_adhesive_cannot_be_bedded():
    m=MaterialInventory()
    with pytest.raises(ValueError,match='No adhesive'):m.bed(0,.003)
    m.spread(0,.0001);m.advance(2401)
    with pytest.raises(ValueError,match='placement window'):m.bed(0,.003)

def test_sponge_removal_requires_contact_and_motion_and_has_finite_capacity():
    s=SurfaceGrout(.1524,capacity_m3=1e-8);before=s.remaining_m3
    for f,v in [(0,.1),(10,0)]:
        s.contact([0,0],[.1,.1],f,v,.1)
        assert s.remaining_m3==before
    out=s.contact([0,0],[.1,.1],10,.1,10)
    assert out['removed_m3']>0 and out['drag_n']>0
    assert s.collected_m3==pytest.approx(s.capacity_m3)
    assert s.remaining_m3+s.collected_m3==pytest.approx(s.initial_m3,abs=1e-14)
    assert s.contact([0,0],[.1,.1],10,.1,1)['removed_m3']==0
    cured=SurfaceGrout(.1524,age_s=10801)
    assert cured.contact([0,0],[.1,.1],10,.1,1)['removed_m3']==0

def test_squeeze_diagnostic_is_rate_sensitive_and_inapplicable_outside_window():
    p=PasteState('adhesive')
    assert squeeze_force(p,.02,.003,0)==0
    assert squeeze_force(p,.02,.002,.001)>squeeze_force(p,.02,.004,.001)
    assert workbench(age_s=2500,exposure_s=2401)['squeeze_force_n'] is None
    for kwargs in [dict(age_s=float('nan')),dict(gap_m=0),dict(exposure_s=1000,age_s=500),dict(speed_m_s=-1)]:
        with pytest.raises(ValueError):workbench(**kwargs)
