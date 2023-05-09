from src.turtlesim import TurtlesimEnvSingle, DqnSingle

env = TurtlesimEnvSingle()  # utworzenie środowiska
env.setup('data/scenario.csv', agent_cnt=1)  # połączenie z symulatorem
env.SPEED_FINE_RATE = -5.0  # zmiana wybranych parametrów środowiska
agents = env.reset()  # ustawienie agenta
tname = list(agents.keys())[0]  # 'lista agentów' do wytrenowania
dqns = DqnSingle(env)  # utworzenie klasy uczącej
dqns.make_model()  # skonstruowanie sieci neuronowej
# dqns.model=load_model('test.h5')                          # albo załadowanie zapisanej wcześniej
dqns.train_main(tname, save_model=True)  # wywołanie uczenia
