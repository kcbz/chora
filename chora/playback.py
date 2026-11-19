"""Wall-clock pacing for live viewing; never changes simulated timesteps."""
from collections import deque

SPEEDS=(1,2,4)

def validate_speed(value):
    if isinstance(value,bool) or not isinstance(value,(int,float)) or value not in SPEEDS:
        raise ValueError('Playback speed must be 1, 2, or 4')
    return value

class PlaybackClock:
    def __init__(self,sim_time,wall_time):
        self.sim_time=sim_time;self.wall_time=wall_time
        self.samples=deque([(wall_time,sim_time)])
    def delay(self,sim_time,wall_time,speed):
        validate_speed(speed)
        return max(0.,(sim_time-self.sim_time)/speed-(wall_time-self.wall_time))
    def commit(self,sim_time,wall_time):
        self.sim_time=sim_time;self.wall_time=wall_time
        self.samples.append((wall_time,sim_time))
        while len(self.samples)>2 and self.samples[1][0]<wall_time-1:self.samples.popleft()
        wall,sim=self.samples[0];elapsed=wall_time-wall
        return (sim_time-sim)/elapsed if elapsed>0 else None
