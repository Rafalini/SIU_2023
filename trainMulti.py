from src.turtlesim import TurtlesimEnvMulti
from src.turtlesim.dqn_multi import DqnMulti
import sys

if __name__ == "__main__":

    env = TurtlesimEnvMulti()                           # utworzenie środowiska
    env.PI_BY = 3                                       # zmiana wybranych parametrów środowiska
    env.SPEED_FINE_RATE = -5.0                          # zmiana wybranych parametrów środowiska
    env.DETECT_COLLISION = True
    env.setup('./data/scenario_C_m.csv')                # połączenie z symulatorem
    agents = env.reset()                                # ustawienie agenta
    dqnm = DqnMulti(env)                                # utworzenie klasy uczącej

    if len(sys.argv) > 1:
        dqnm.load_model(path=sys.argv[1])           # albo załadowanie zapisanej wcześniej
    else:
        dqnm.make_model()                           # skonstruowanie sieci neuronowej

    dqnm.train_main(save_model=True, save_state=True)   # wywołanie uczenia (wyniki zapisywane okresowo)