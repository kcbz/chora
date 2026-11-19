"""Conservation/contact checks for the declared reduced mortar model."""
import numpy as np
import pytest
from chora.mortar_layer import MortarLayer
from chora.tile_installation import TARGETS


def test_metering_requires_motion_near_substrate_and_conserves_material():
    m=MortarLayer(TARGETS)
    assert m.arm_dose(94)=='dose_armed_for_tool_motion'
    initial=m.reservoir_m3
    p=np.array([.43,.02,.3])
    m.apply_tool(p-[.001,0,0],p,.02)
    p[2]=.2408;m.apply_tool(p,p,.02)
    assert m.reservoir_m3==initial
    for x in np.linspace(.3488,.5112,1000):
        p=np.array([x,.02,.2408]);m.apply_tool(p-[.0001624,0,0],p,.01)
    assert 90<m.applied_by_tool_m3*1e6<94
    assert m.volume[0].sum()>0 and not m.volume[1:].any()
    assert m.pending_m3<4e-6
    assert abs(m.balance_error_m3)<1e-14
    m.arm_dose(0);assert m.pending_m3==0
    m.paste.advance(7201)
    assert m.arm_dose(1)=='adhesive_expired'


def test_bedding_forces_depend_on_contact_and_redistribution_preserves_volume():
    m=MortarLayer(TARGETS)
    m.volume[0,::2]=.008*m.area
    m.reservoir_m3-=m.volume.sum()
    before=m.volume.sum()
    R=np.eye(3);zero=np.zeros(3)
    f,t=m.tile_forces(0,np.array([.43,.02,.30]),R,zero,zero,.001)
    assert np.linalg.norm(f)==0
    # Bottom at 4 mm above substrate: ridges load; troughs are initially empty.
    for _ in range(100):
        f,t=m.tile_forces(0,np.array([.43,.02,.25035]),R,np.array([0,0,-.001]),zero,.001)
    assert f[2]>0 and t[0]<0
    assert np.linalg.norm(t)<f[2]*m.width/2
    assert m.redistributed_m3>0 and np.count_nonzero(m.volume[0])>128
    assert m.volume.sum()<=before+1e-15
    assert abs(m.balance_error_m3)<1e-14
    assert np.all(m.volume>=0)


def test_contact_tracks_world_geometry_instead_of_semantic_tile_assignment():
    m=MortarLayer(TARGETS);m.volume[0,:]=.004*m.area
    m.reservoir_m3-=m.volume.sum();m.advance(.001)
    # No tile ID is supplied: any free tile at patch 0 must receive support.
    f,t=m.couple_tile(np.array([.43,.02,.24935]),np.eye(3),np.zeros(3),np.zeros(3),.001)
    assert f[2]>0 and m.loaded_n[0]>0
    m.advance(.001)
    f,t=m.couple_tile(np.array([.591925,.181925,.24935]),np.eye(3),np.zeros(3),np.zeros(3),.001)
    assert np.linalg.norm(f)==0 and not m.loaded_n.any()
