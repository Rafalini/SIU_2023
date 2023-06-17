from src.turtlesim import TurtlesimEnvSingle, DqnSingle
import sys

# print("Usage: train.py <scenario_path> <model_path>")

if __name__ == "__main__":

    env = TurtlesimEnvSingle()  # utworzenie środowiska

    if len(sys.argv) > 1:
        env.setup(path=sys.argv[1], agent_cnt=1)  # połączenie z symulatorem
    else:
        env.setup('data/scenario_C_px.csv', agent_cnt=1)  # połączenie z symulatorem


    agents = env.reset()  # ustawienie agenta
    tname = list(agents.keys())[0]  # 'lista agentów' do wytrenowania
    dqns = DqnSingle(env)  # utworzenie klasy uczącej


    if len(sys.argv) > 2:
        dqns.load_model(path=sys.argv[2])    # albo załadowanie zapisanej wcześniej
    else:
        dqns.make_model()  # skonstruowanie sieci neuronowej

    dqns.train_main(tname, save_model=True)  # wywołanie uczenia
