import sys
import numpy as np
from tensorflow import keras
from copy import deepcopy
from run import SimulationRunner
from src.turtlesim.turtlesim_env_single import TurtlesimEnvSingle


class SimulationRunnerMulti(SimulationRunner):

    def __init__(self, model_path: str, scenario_path: str):
        super().__init__(model_path, scenario_path)

    # złożenie dwóch rastrów sytuacji aktualnej i poprzedniej w tensor 5x5x10 wejścia do sieci
    def _inp_stack(self, last, cur):
        # fa,fd,fc+1,fp+1 - z wyjścia get_map - BEZ 2 POCZ. WARTOŚCI (zalecana prędkość w ukł. odniesienia planszy)
        inp = np.stack([cur[2], cur[3], cur[4], cur[5], cur[6], last[2], last[3], last[4], last[5], last[6]], axis=-1)
        return inp

    def run_simulation(self):
        env = TurtlesimEnvSingle()
        env.setup(self.scenario_path, agent_cnt=8)
        agents = env.reset()
        current_states = {}
        for agent in agents.keys():
            current_states[agent] = deepcopy(agents[agent].map)

        while not env.out_of_track:
            for agent in agents.keys():
                last_state = deepcopy(current_states[agent])
                control = np.argmax(self._decision(self.model, last_state, current_states[agent]))
                current_states[agent], _, _ = env.step({agent: self._ctl_2_act(control)})  # noqa


if __name__ == "__main__":
    if len(sys.argv) > 2:
        SimulationRunnerMulti(model_path=sys.argv[1], scenario_path=sys.argv[2]).run_simulation()
    else:
        print("Missing model or scenario path!. Usage: run.py <model_path> <scenario_path>")

