from dataclasses import dataclass


@dataclass
class Sequence:
	route_id: int
	agent_count: int
	xmin: float
	xmax: float
	ymin: float
	ymax: float
	xg: float
	yg: float

	def __truediv__(self, scalar):
		return Sequence(route_id=self.route_id,
						agent_count=self.agent_count,
						xmin=self.xmin / scalar,
						xmax=self.xmax / scalar,
						ymin=self.ymin / scalar,
						ymax=self.ymax / scalar,
						xg=self.xg / scalar,
						yg=self.yg / scalar)

	def __mul__(self, scalar):
		return Sequence(route_id=self.route_id,
						agent_count=self.agent_count,
						xmin=self.xmin * scalar,
						xmax=self.xmax * scalar,
						ymin=self.ymin * scalar,
						ymax=self.ymax * scalar,
						xg=self.xg * scalar,
						yg=self.yg * scalar)
	__rmul__ = __mul__

	def get_coords(self) -> list:
		return [self.xmin, self.xmax, self.ymin, self.ymax, self.xg, self.yg]
