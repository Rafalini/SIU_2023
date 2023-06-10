import sys
import numpy as np
from tensorflow import keras
from copy import deepcopy

from src.turtlesim.turtlesim_env_single import TurtlesimEnvSingle


class SimulationRunner:
    def __init__(self, model_path: str, scenario_path: str):
        self.model = keras.models.load_model(model_path)
        self.scenario_path = scenario_path

    def _decision(self, the_model, last, cur):
        inp = np.expand_dims(self._inp_stack(last, cur), axis=-1)
        inp = np.expand_dims(inp, axis=0)
        return the_model(inp).numpy().flatten()

    # zakodowanie wybranego sterowania (0-5) na potrzeby środowiska: (prędkość,skręt)
    def _ctl_2_act(self, decision: int):  # prędkość\skręt    -.1rad 0 .1rad
        v = .2             # 0.2                0   1   2
        if decision >= 3:  # 0.4                3   4   5
            v = .4
        w = .25 * (decision % 3 - 1)
        return [v, w]

    # złożenie dwóch rastrów sytuacji aktualnej i poprzedniej w tensor 5x5x8 wejścia do sieci
    def _inp_stack(self, last, cur):
        # fa,fd,fc+1,fp+1 - z wyjścia get_map - BEZ 2 POCZ. WARTOŚCI (zalecana prędkość w ukł. odniesienia planszy)
        inp = np.stack([cur[2], cur[3], cur[4], cur[5], last[2], last[3], last[4], last[5]], axis=-1)
        return inp

    def run_simulation(self):
        env = TurtlesimEnvSingle()
        env.setup(self.scenario_path, agent_cnt=1)
        agents = env.reset()
        tname = list(agents.keys())[0]
        current_state = deepcopy(agents[tname].map)
        print(current_state)
        while not env.out_of_track:
            last_state = deepcopy(current_state)
            control = np.argmax(self._decision(self.model, last_state, current_state))
            current_state, _, _ = env.step({tname: self._ctl_2_act(control)})  # noqa


if __name__ == "__main__":
    if len(sys.argv) > 2:
        SimulationRunner(model_path=sys.argv[1], scenario_path=sys.argv[2]).run_simulation()
    else:
        print("Missing model or scenario path!. Usage: run.py <model_path> <scenario_path>")

