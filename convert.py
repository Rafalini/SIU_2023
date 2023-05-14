from src.turtlesim import TurtlesimEnvSingle, DqnSingle
import sys

env = TurtlesimEnvSingle()
dqns = DqnSingle(env)
dqns.load_model(path = sys.argv[1])
dqns.model.save(f"models/model-{sys.argv[1]}.tf", save_format="tf")
