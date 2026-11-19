import pytest
from chora.playback import PlaybackClock,validate_speed

@pytest.mark.parametrize('speed,expected',[(1,.08),(2,.03),(4,.005)])
def test_pacing_subtracts_all_computation_and_render_time(speed,expected):
    c=PlaybackClock(10.,20.)
    assert c.delay(10.1,20.02,speed)==pytest.approx(expected)

def test_overloaded_viewer_does_not_sleep_or_claim_requested_speed():
    c=PlaybackClock(0,0)
    assert c.delay(.05,.2,4)==0
    assert c.commit(.05,.2)==pytest.approx(.25)
    assert c.delay(.1,.21,1)==pytest.approx(.04) # speed changes mid-run

@pytest.mark.parametrize('speed',[True,None,'4',0,3,-1,float('nan'),float('inf')])
def test_rejects_invalid_speeds(speed):
    with pytest.raises(ValueError):validate_speed(speed)
