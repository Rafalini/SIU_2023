from src.turtlesim import TurtlesimEnvSingle
from src.turtlesim.dqn_multi import DqnMulti
import sys

if __name__ == "__main__":

    env = TurtlesimEnvSingle()  # utworzenie środowiska
    env.setup('data/scenario_C_px.csv', agent_cnt=1)  # połączenie z symulatorem
    env.SPEED_FINE_RATE = -5.0  # zmiana wybranych parametrów środowiska
    agents = env.reset()  # ustawienie agenta
    tname = list(agents.keys())[0]  # 'lista agentów' do wytrenowania
    dqnm = DqnMulti(env)  # utworzenie klasy uczącej

    if len(sys.argv) > 1:
        dqnm.load_model(path = sys.argv[1])    # albo załadowanie zapisanej wcześniej
    else:
        dqnm.make_model()  # skonstruowanie sieci neuronowej

    dqnm.train_main(save_model=True, save_state=True)