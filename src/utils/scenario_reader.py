import csv
import enum
from typing import Optional
from .sequence import Sequence


class UNIT(enum.Enum):
	PIXELS = 0
	METERS = 1


class ScenarioReader:
	"""
	Helper class for loading scenario and converting scenario unit (pixels/meters)
	"""
	def __init__(self, scenario_file: str, px_meter_ratio: float):
		"""
		Class constructor
		Args:
	 		scenario_file (str): path to CSV file containing scenario
	 		px_meter_ratio (float): pixel to meter ratio (for conversion purposes)
		"""
		self.scenario_file = scenario_file
		self.px_meter_ratio = px_meter_ratio
		self.distance_unit = UNIT.PIXELS
		self.sequences = []
		self._load_file()

	def _load_file(self):
		"""
		Method for loading CSV file content into list object
		"""
		with open(self.scenario_file, encoding='utf-8') as file:  # załadowanie tras agentów
			scenario = csv.reader(file, delimiter=';')
			self.sequences = [
				Sequence(route_id=int(s[0]),
						 agent_count=int(s[1]),
						 xmin=float(s[2]),
						 xmax=float(s[3]),
						 ymin=float(s[4]),
						 ymax=float(s[5]),
						 xg=float(s[6]),
						 yg=float(s[7])
						 ) for s in scenario
			]

	def to_meters(self):
		"""
		Method for converting (inplace) sequence values to meters
		"""
		if self.distance_unit == UNIT.METERS:
			return self

		self.sequences = [sequence / self.px_meter_ratio for sequence in self.sequences]
		self.distance_unit = UNIT.PIXELS

		return self

	def to_pixels(self):
		"""
			Method for converting (inplace) sequence values to pixels
			"""
		if self.distance_unit == UNIT.PIXELS:
			return self

		self.sequences = [sequence * self.px_meter_ratio for sequence in self.sequences]
		self.distance_unit = UNIT.METERS

		return self

	def get_sequences(self, route_id: Optional[int] = None) -> list:
		"""
		Method for getting sequences for given route (or all routes)
		Args:
	 		route_id (int): identifier of the route
	 	Returns:
			list: list of sequences coordinates (as a list of 6 integers representing xmin-xmax-ymin-ymax-xg-yg)
		"""
		if route_id is None:
			return [sequence.get_coords() for sequence in self.sequences]

		return [sequence.get_coords() for sequence in self.sequences if sequence.route_id == route_id]

	def get_routes(self):
		"""
		Method for getting routes in format required by turtlesim library
		Returns:
			 dict: dict of routes, where key is route id and value is a list of
			  7 integers representing agent_count-xmin-xmax-ymin-ymax-xg-yg
		"""
		routes = {}
		for sequence in self.sequences:
			if sequence.route_id not in routes:
				routes[sequence.route_id] = []
			routes[sequence.route_id].append([sequence.agent_count, *sequence.get_coords()])
		return routes
