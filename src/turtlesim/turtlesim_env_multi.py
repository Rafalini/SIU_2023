import numpy as np
import rospy
from turtlesim.msg import Pose
from geometry_msgs.msg import Twist

from turtlesim_env_base import TurtlesimEnvBase


class TurtlesimEnvMulti(TurtlesimEnvBase):
    def __init__(self):
        super().__init__()

    def setup(self, routes_fname: str, agent_cnt=None):
        super().setup(routes_fname, agent_cnt)
        for agent in self.agents.values():  # liczba kroków - indywidualnie dla każdego agenta
            agent.step_sum = 0

    def reset(self, tnames=None, sections='default'):
        ret = super().reset(tnames, sections)
        if tnames is None:
            tnames = self.agents.keys()
        for tname in tnames:
            self.agents[tname].step_sum = 0  # liczba kroków zerowana wybiórczo
        return ret

    def _make_smooth_step(self, tname: str, speed: float, turn: float):
        """
        Helper method for making smooth step (forward, turn, forward, turn)
        Args:
            tname (str): Turtle name
            speed (float): forward movement
            turn (float): left rotation
        """
        twist = Twist()
        twist.linear.x = speed * self.px_meter_ratio / 4
        twist.angular.z = 0
        self.tapi.setVel(tname, twist)

        twist = Twist()
        twist.linear.x = speed * self.px_meter_ratio / 4
        twist.angular.z = turn * self.px_meter_ratio
        self.tapi.setVel(tname, twist)

        twist = Twist()
        twist.linear.x = speed * self.px_meter_ratio / 4
        twist.angular.z = 0
        self.tapi.setVel(tname, twist)

        twist = Twist()
        twist.linear.x = speed * self.px_meter_ratio / 4
        twist.angular.z = turn * self.px_meter_ratio
        self.tapi.setVel(tname, twist)

    def _make_step(self, tname: str, pose: Pose, speed: float, turn: float):
        """
        Helper method for making regular step (jump)
        Args:
            tname (str): Turtle name
            pose (Pose): initial position (before making the step)
            speed (float): forward movement
            turn (float): left rotation
        """
        vx = np.cos(pose.theta + turn) * speed * self.SEC_PER_STEP
        vy = np.sin(pose.theta + turn) * speed * self.SEC_PER_STEP
        p = Pose(x=pose.x + vx, y=pose.y + vy, theta=pose.theta + turn)
        self.tapi.setPose(tname, p, mode='absolute')
        rospy.sleep(self.WAIT_AFTER_MOVE)
    def step(self, actions, realtime=False):  # {id_żółwia:(prędkość,skręt)}
        # pozycja PRZED krokiem sterowania
        for tname, action in actions.items():
            agent = self.agents[tname]
            agent.step_sum += 1
            agent.pose = self.tapi.getPose(tname)  # zapamiętanie położenia przed wykonaniem ruchu
            _, _, _, agent.fd, _, _ = self.get_road(tname)  # odl. do celu (na wypadek, gdyby uległa zmianie)
        # action: [prędkość,skręt]
        # TODO STUDENCI przejechać 1/2 okresu, skręcić, przejechać pozostałą 1/2   chyba DONE
        if realtime:  # jazda+skręt+jazda
            # twist = Twist()
            # ...
            # self.tapi.setVel(tname, twist)
            # ...
            action = list(actions.values())[0]  # uwzględniamy 1 akcje
            tname = list(self.agents.keys())[0]  # sterujemy 1 żółwiem
            init_pose = self.tapi.getPose(tname)  # pozycja PRZED krokiem sterowania
            if realtime:  # jazda+skręt+jazda+skręt
                self._make_smooth_step(tname=tname, speed=action[0], turn=action[1])
            else:  # skok+obrót
                self._make_step(tname=tname, pose=init_pose, speed=action[0], turn=action[1])
        else:  # skok+obrót
            for tname, action in actions.items():
                pose = self.agents[tname].pose
                # obliczenie i wykonanie przesunięcia
                vx = np.cos(pose.theta + action[1]) * action[0] * self.SEC_PER_STEP
                vy = np.sin(pose.theta + action[1]) * action[0] * self.SEC_PER_STEP
                p = Pose(x=pose.x + vx, y=pose.y + vy, theta=pose.theta + action[1])
                self.tapi.setPose(tname, p, mode='absolute')
            rospy.sleep(self.WAIT_AFTER_MOVE)
        # pozycje i sytuacje PO kroku sterowania
        ret = {}  # {tname:(get_map(),reward,done)}
        collisions = self.tapi.getColisions(self.agents.keys(), self.COLLISION_DIST)
        colliding = set()  # nazwy kolidujących agentów
        for collision in collisions:
            colliding.add(collision['name1'])
            colliding.add(collision['name2'])
        for tname in actions:
            pose = self.agents[tname].pose  # położenie przed ruchem
            pose1 = self.tapi.getPose(tname)  # położenie po ruchu
            self.agents[tname].pose = pose1  # TODO nowa linia -> studenci

            fx1, fy1, fa1, fd1, _, _ = self.get_road(tname)  # warunki drogowe po przemieszczeniu
            vx1 = (pose1.x - pose.x) / self.SEC_PER_STEP  # aktualna prędkość - składowa x
            vy1 = (pose1.y - pose.y) / self.SEC_PER_STEP  # -"-                   y
            v1 = np.sqrt(vx1 ** 2 + vy1 ** 2)  # aktualny moduł prędkości
            fv1 = np.sqrt(fx1 ** 2 + fy1 ** 2)  # zalecany moduł prędkości
            # wyznaczenie składników funkcji celu
            done = False
            r1 = min(0, self.SPEED_FINE_RATE * (v1 - fv1))  # kara za przekroczenie prędkości
            r2 = 0
            if fv1 > .001:
                vf1 = (vx1 * fx1 + vy1 * fy1) / fv1  # rzut prędkości faktycznej na zalecaną
                if vf1 > 0:
                    r2 = self.SPEED_RWRD_RATE * vf1  # nagroda za jazdę z prądem
                else:
                    r2 = -self.SPEED_RVRS_RATE * vf1  # kara za jazdę pod prąd
            r3 = self.DIST_RWRD_RATE * (self.agents[tname].fd - fd1)  # nagroda za zbliżenie się do celu
            r4 = 0
            if abs(fx1) + abs(fy1) < .01 and fa1 == 1:  # wylądowaliśmy poza trasą
                r4 = self.OUT_OF_TRACK_FINE
                done = True
            map = self.get_map(tname)
            fo = map[6]
            # wykrywanie kolizji
            r5 = 0
            if self.DETECT_COLLISION and fo[self.GRID_RES // 2, self.GRID_RES - 1] == 0 and tname in colliding:
                r5 = self.OUT_OF_TRACK_FINE
                done = True
            reward = fa1 * (r1 + r2) + r3 + r4 + r5
            # sp=speed, fl=flow, cl=closing, tr=track, xx=collision
            # print(f'RWD: {reward:.2f} = {fa1:.2f}*(sp{r1:.2f} fl{r2:.2f}) cl{r3:.2f} tr{r4:.2f} xx{r5:.2f}')
            if self.agents[tname].step_sum > self.MAX_STEPS:
                done = True
            ret[tname] = (map, reward, done)
        return ret


def provide_env():
    return TurtlesimEnvMulti()


if __name__ == "__main__":
    import random

    env = provide_env()
    env.PI_BY = 100  # początkowo wszyscy skierowani praktycznie na azymut
    env.DETECT_COLLISION = True
    env.setup('routes.csv', agent_cnt=100)
    agents = env.reset()
    for i in range(100):  # losowy agent wykonuje losowy ruch
        tname = random.choice(list(agents.keys()))
        res = env.step({tname: (random.uniform(.2, 1), random.uniform(-.3, .3))})
        if res[tname][2]:  # kolizja lub aut wyklucza agenta z dalszej symulacji
            del agents[tname]
