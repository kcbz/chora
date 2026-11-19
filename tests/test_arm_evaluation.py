import numpy as np
import mujoco
from chora.arm_evaluation import build_spec,MODELS,TASKS,torque_limits


def test_screen_does_not_treat_unbounded_model_as_infinite_strength():
    model=build_spec('iiwa14').compile()
    assert not np.any(np.isfinite(torque_limits(model)))
    model=build_spec('fr3').compile()
    assert np.allclose(torque_limits(model),[87]*4+[12]*3)


def test_point_payload_gravity_matches_explicit_native_payload():
    spec=build_spec('fr3');bare=spec.compile();data=mujoco.MjData(bare)
    mujoco.mj_resetDataKeyframe(bare,data,0);mujoco.mj_forward(bare,data)
    sid=bare.site('eval_flange').id;body=int(bare.site_bodyid[sid]);offset=np.array(TASKS['cutting']['com']);mass=TASKS['cutting']['mass']
    com=data.site_xpos[sid]+data.site_xmat[sid].reshape(3,3)@offset
    J=np.zeros((3,bare.nv));mujoco.mj_jac(bare,data,J,None,com,body)
    expected=-J.T@np.array([0,0,-mass*9.81])
    flange=spec.site('eval_flange');R=np.zeros(9);mujoco.mju_quat2Mat(R,flange.quat)
    flange.parent.add_body(name='test_payload',pos=np.array(flange.pos)+R.reshape(3,3)@offset,mass=mass,ipos=[0,0,0],inertia=[.001]*3,explicitinertial=True)
    loaded=spec.compile();ld=mujoco.MjData(loaded);ld.qpos[:]=data.qpos;mujoco.mj_forward(loaded,ld)
    assert np.allclose(ld.qfrc_bias-data.qfrc_bias,expected,atol=1e-9)
