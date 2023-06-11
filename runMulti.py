import sys
import numpy as np
from tensorflow import keras
from copy import deepcopy
from run import SimulationRunner
from src.turtlesim.turtlesim_env_multi import TurtlesimEnvMulti


class SimulationRunnerMulti(SimulationRunner):

    def __init__(self, model_path: str, scenario_path: str):
        super().__init__(model_path, scenario_path)

    # złożenie dwóch rastrów sytuacji aktualnej i poprzedniej w tensor 5x5x10 wejścia do sieci
    def _inp_stack(self, last, cur):
        # fa,fd,fc+1,fp+1 - z wyjścia get_map - BEZ 2 POCZ. WARTOŚCI (zalecana prędkość w ukł. odniesienia planszy)
        inp = np.stack([cur[2], cur[3], cur[4], cur[5], cur[6], last[2], last[3], last[4], last[5], last[6]], axis=-1)
        return inp

    def run_simulation(self):
        env = TurtlesimEnvMulti()
        env.setup(self.scenario_path, agent_cnt=4)
        env.reset()
        current_states = {tname: agent.map for tname, agent in env.agents.items()}  # aktualne sytuacje
        last_states = {tname: agent.map for tname, agent in env.agents.items()}

        while not env.out_of_track:
            controls = {}
            for tname in env.agents:
                controls[tname] = np.argmax(self._decision(self.model, last_states[tname], current_states[tname]))
            actions = {tname: self._ctl_2_act(control) for tname, control in controls.items()}
            scene = env.step(actions)
            for tname, (new_state, reward, done) in scene.items():
                last_states[tname] = current_states[tname]
                current_states[tname] = new_state

if __name__ == "__main__":
    if len(sys.argv) > 2:
        SimulationRunnerMulti(model_path=sys.argv[1], scenario_path=sys.argv[2]).run_simulation()
    else:
        print("Missing model or scenario path!. Usage: runMulti.py <model_path> <scenario_path>")

