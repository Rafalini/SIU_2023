# encoding: utf8
import numpy as np
import rospy
from turtlesim.msg import Pose
from geometry_msgs.msg import Twist

from turtlesim_env_base import TurtlesimEnvBase, TurtleAgent

class TurtlesimEnvSingle(TurtlesimEnvBase):
    def __init__(self):
        super().__init__()
    def step(self,actions,realtime=False):
        self.step_sum += 1
        action=list(actions.values())[0]                # uwzględniamy 1. (i jedyną akcję w słowniku)
        tname=list(self.agents.keys())[0]               # sterujemy 1. (i jedynym) żółwiem
        # pozycja PRZED krokiem sterowania
        pose=self.tapi.getPose(tname)
        _,_,_,fd,_,_ = self.get_road(tname)             # odl. do celu (mógł ulec zmianie)
        # action: [prędkość,skręt]
        # TODO STUDENCI przejechać 1/2 okresu, skręcić, przejechać pozostałą 1/2
        if realtime:                                    # jazda+skręt+jazda+skręt
            twist = Twist()
            ...
            self.tapi.setVel(tname,twist)
            ...
        else:                                           # skok+obrót
            # obliczenie i wykonanie przesunięcia
            vx = np.cos(pose.theta+action[1])*action[0]*self.SEC_PER_STEP
            vy = np.sin(pose.theta+action[1])*action[0]*self.SEC_PER_STEP
            p=Pose(x=pose.x+vx,y=pose.y+vy,theta=pose.theta+action[1])
            self.tapi.setPose(tname,p,mode='absolute')
            rospy.sleep(self.WAIT_AFTER_MOVE)
        # pozycja PO kroku sterowania
        done=False                                      # flaga wykrytego końca scenariusza symulacji
        pose1 = self.tapi.getPose(tname)
        self.agents[tname].pose=pose1
        fx1,fy1,fa1,fd1,_,_ = self.get_road(tname)      # warunki drogowe po przemieszczeniu
        vx1 = (pose1.x-pose.x)/self.SEC_PER_STEP        # aktualna prędkość - składowa x
        vy1 = (pose1.y-pose.y)/self.SEC_PER_STEP        #        -"-                   y
        v1  = np.sqrt(vx1**2+vy1**2)                    # aktualny moduł prędkości
        fv1 = np.sqrt(fx1**2+fy1**2)                    # zalecany moduł prędkości
        # wyznaczenie składników funkcji celu
        r1 = min(0,self.SPEED_FINE_RATE*(v1-fv1))       # kara za przekroczenie prędkości
        r2 = 0
        if fv1>.001:
            vf1 = (vx1*fx1+vy1*fy1)/fv1                 # rzut prędkości faktycznej na zalecaną
            if vf1>0:
                r2 = self.SPEED_RWRD_RATE*vf1           # nagroda za jazdę z prądem
            else:
                r2 = -self.SPEED_RVRS_RATE*vf1          # kara za jazdę pod prąd
        r3 = self.DIST_RWRD_RATE*(fd-fd1)               # nagroda za zbliżenie się do celu
        r4=0
        if abs(fx1)+abs(fy1)<.01 and fa1==1:            # wylądowaliśmy poza trasą
            r4 = self.OUT_OF_TRACK_FINE
            done=True
        reward=fa1*(r1+r2)+r3+r4
        # sp=speed, fl=flow, cl=closing, tr=track
        # print(f'RWD: {reward:.2f} = {fa1:.2f}*(sp{r1:.2f} fl{r2:.2f}) cl{r3:.2f} tr{r4:.2f}')
        if self.step_sum>self.MAX_STEPS:
            done=True
        return self.get_map(tname),reward,done

def provide_env():
    return TurtlesimEnvSingle()

# 10 kroków 1 żółwia z losowego segmentu, z losową prędkością <0,2;1> i skrętem <-0,3;0,3>
if __name__ == "__main__":
    import random
    env=provide_env()
    env.setup('routes.csv',agent_cnt=1)
    agents=env.reset()
    tname=list(agents.keys())[0]
    for i in range(10):                                     # ruch losowy do przodu
        env.step({tname:(random.uniform(.2,1),random.uniform(-.3,.3))},realtime=False)
