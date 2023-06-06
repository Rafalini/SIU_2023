from src.turtlesim import TurtlesimEnvSingle, DqnSingle
import sys

if __name__ == "__main__":

    env = TurtlesimEnvSingle()  # utworzenie środowiska
    env.setup('data/scenario_C_m.csv', agent_cnt=1)  # połączenie z symulatorem
    env.SPEED_FINE_RATE = -5.0  # zmiana wybranych parametrów środowiska
    agents = env.reset()  # ustawienie agenta
    tname = list(agents.keys())[0]  # 'lista agentów' do wytrenowania
    dqns = DqnSingle(env)  # utworzenie klasy uczącej

    if len(sys.argv) > 1:
        dqns.load_model(path=sys.argv[1])    # albo załadowanie zapisanej wcześniej
    else:
        dqns.make_model()  # skonstruowanie sieci neuronowej

    dqns.train_main(tname, save_model=True)  # wywołanie uczenia
