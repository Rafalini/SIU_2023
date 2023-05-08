from typing import Tuple, Dict

import numpy as np
import rospy
from geometry_msgs.msg import Twist
from turtlesim.msg import Pose

from .turtlesim_env_base import TurtlesimEnvBase


class TurtlesimEnvSingle(TurtlesimEnvBase):
	def __init__(self):
		super().__init__()
		self.done = False

	def step(self,
			 actions: Dict[str, Tuple[float, float]],
			 realtime: bool = False) -> Tuple[Tuple[float, float, float, float, float, float, float], float, bool]:
		"""
		Implementation of baseclass abstract method for making a step
		Args:
			actions (Dict[str, Tuple[float, float]]): list of actions
			realtime (bool): if true, simulate smooth movement, else make a jump-move
		Returns:
			(float, float, float, float, float, float, float), float, bool: map tuple, step reward, simululation status
		"""
		self.step_sum += 1
		action = list(actions.values())[0]  # uwzględniamy 1. (i jedyną akcję w słowniku)
		tname = list(self.agents.keys())[0]  # sterujemy 1. (i jedynym) żółwiem
		init_pose = self.tapi.getPose(tname)	# pozycja PRZED krokiem sterowania
		_, _, _, fd, _, _ = self.get_road(tname)  # odl. do celu (mógł ulec zmianie)

		if realtime:  # jazda+skręt+jazda+skręt
			self._make_smooth_step(tname=tname, speed=action[0], turn=action[1])
		else:  # skok+obrót
			self._make_step(tname=tname, pose=init_pose, speed=action[0], turn=action[1])

		final_pose = self.tapi.getPose(tname)	# pozycja PO kroku sterowania
		self.agents[tname].pose = final_pose

		reward = self._calculate_reward(tname=tname, init_pose=init_pose, final_pose=final_pose, fd=fd)

		if self.step_sum > self.MAX_STEPS:
			self.done = True

		return self.get_map(tname), reward, self.done

	def _calculate_reward(self, tname: str, init_pose: Pose, final_pose: Pose, fd: float):
		"""
		Helper method for calculating step reward
		Args:
			tname (str): Turtle name 
			init_pose (Pose): initial position (before making the step)
			final_pose (Pose): final position (afet making the step)
			fd (float): distance to target
		Returns:
			float: reward fot making the step
		"""
		fx1, fy1, fa1, fd1, _, _ = self.get_road(tname)  # warunki drogowe po przemieszczeniu
		vx1 = (final_pose.x - init_pose.x) / self.SEC_PER_STEP  # aktualna prędkość - składowa x
		vy1 = (final_pose.y - init_pose.y) / self.SEC_PER_STEP  # aktualna prędkość - składowa y
		v1 = np.sqrt(vx1 ** 2 + vy1 ** 2)  # aktualny moduł prędkości
		fv1 = np.sqrt(fx1 ** 2 + fy1 ** 2)  # zalecany moduł prędkości
		# wyznaczenie składników funkcji celu
		r1 = min(0, self.SPEED_FINE_RATE * (v1 - fv1))  # kara za przekroczenie prędkości
		r2 = 0
		if fv1 > .001:
			vf1 = (vx1 * fx1 + vy1 * fy1) / fv1  # rzut prędkości faktycznej na zalecaną
			if vf1 > 0:
				r2 = self.SPEED_RWRD_RATE * vf1  # nagroda za jazdę z prądem
			else:
				r2 = -self.SPEED_RVRS_RATE * vf1  # kara za jazdę pod prąd
		r3 = self.DIST_RWRD_RATE * (fd - fd1)  # nagroda za zbliżenie się do celu
		r4 = 0

		if abs(fx1) + abs(fy1) < .01 and fa1 == 1:  # wylądowaliśmy poza trasą
			r4 = self.OUT_OF_TRACK_FINE
			self.done = True

		reward = fa1 * (r1 + r2) + r3 + r4

		# sp=speed, fl=flow, cl=closing, tr=track
		# print(f'RWD: {reward:.2f} = {fa1:.2f}*(sp{r1:.2f} fl{r2:.2f}) cl{r3:.2f} tr{r4:.2f}')

		return reward

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
