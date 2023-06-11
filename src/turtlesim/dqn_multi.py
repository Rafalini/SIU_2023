import pickle
import numpy as np
from collections import deque
import tensorflow as tf
tf.get_logger().setLevel(3)
from tensorflow import keras

from datetime import datetime
from keras.models import Sequential
from keras.layers import *
from .turtlesim_env_base import TurtlesimEnvBase
from .dqn_single import DqnSingle

class DqnMulti(DqnSingle):
    def __init__(self,env:TurtlesimEnvBase,id_prefix='dqnm',seed=42):
        super().__init__(env,id_prefix,seed)
        self.SAVE_MODEL_EVERY = 250
    # złożenie dwóch rastrów sytuacji aktualnej i poprzedniej w tensor 5x5x10 wejścia do sieci
    def inp_stack(_,last,cur):
        # fa,fd,fc+1,fp+1 ORAZ fo doklejone na końcu
        inp = np.stack([cur[2],cur[3],cur[4],cur[5],last[2],last[3],last[4],last[5],cur[6],last[6]], axis=-1)
        return inp
    # predykcja nagród łącznych (Q) za sterowania na podst. bieżącej i ostatniej sytuacji
    # wytworzenie modelu - sieci neuronowej
    def make_model(self):
        N=self.env.GRID_RES                                                         # rozdzielczość rastra
        M=10                                                                         # liczba warstw z inp_stack()
        self.model=Sequential()
        self.model.add(Conv3D(filters=2*M,kernel_size=(2,2,M),activation='relu',input_shape=(N,N,M,1)))
        self.model.add(Permute((1,2,4,3)))
        self.model.add(Conv3D(filters=2*M,kernel_size=(2,2,2*M),activation='relu'))
        self.model.add(Permute((1,2,4,3)))
        self.model.add(Conv3D(filters=2*M,kernel_size=(2,2,2*M),activation='relu'))
        self.model.add(Flatten())
        self.model.add(Dense(64,activation='relu'))                                 # (128)
        self.model.add(Dense(self.CTL_DIM,activation="linear"))                     # wyjście Q dla każdej z CTL_DIM decyzji
        self.model.compile(loss='mse',optimizer=keras.optimizers.Adam(learning_rate=0.001),metrics=["accuracy"])
    # model z osobną gałęzią dla logiki unikania kolizji
    def train_main(self,save_model=True,save_state=True):
        self.target_model=keras.models.clone_model(self.model)                      # model pomocniczy (wolnozmienny)
        self.replay_memory=deque(maxlen=self.REPLAY_MEM_SIZE_MAX)                   # historia kroków
        episode_rewards=np.zeros(self.EPISODES_MAX)*np.nan                          # historia nagród w epizodach
        epsilon=self.EPS_INIT
        step_cnt=0
        train_cnt=0
        current_states={tname:agent.map for tname,agent in self.env.agents.items()}     # aktualne sytuacje
        last_states={tname:agent.map for tname,agent in self.env.agents.items()}        # poprzednie stytuacje początkowo takie same
        agent_episode={tname:i for i,tname in enumerate(self.env.agents)}               # indeks epizodu przypisany do agenta
        episode_rewards[:len(self.env.agents)]=0                                        # inicjalizacja nagród za epizody
        episode=len(self.env.agents)-1                                                  # indeks ost. epizodu
        to_restart=set()                                                                # agenty do reaktywacji
        while episode<self.EPISODES_MAX:                                                # ucz w epizodach treningowych
            print(f"Episode: {episode}")
            if episode%self.SAVE_MODEL_EVERY==0:                     # zapisuj co 250 epizodów gdy jest ustawiona flaga
                # current_timestamp_ms = round(time() * 1000)
                current_timestamp_ms = datetime.now().strftime("%d_%m__%H_%M_%S")
                self.model.save(f"models/model-E{episode}-{current_timestamp_ms}.tf", save_format="tf")  # zapisz model w formacie h5
                with open(f"models/model-E{episode}-{current_timestamp_ms}.config", "w+") as config_file:
                    config_file.write(self.xid())

            self.env.reset(to_restart,['random' for i in to_restart])                   # inicjalizacja wybranych
            for tname in to_restart:                                                    # odczytanie sytuacji
                current_states[tname]=self.env.agents[tname].map                        # początkowa sytuacja
                last_states[tname]=[i.copy() for i in current_states[tname]]            # zaczyna od postoju: poprz. stan taki jak obecny
                episode+=1                                                              # dla niego to nowy epizod
                episode_rewards[episode]=0                                              # inicjalizacja nagród w tym epizodzie
                agent_episode[tname]=episode                                            # przypisanie agenta do epizodu
                if (episode+1)%self.SAVE_MODEL_EVERY==0 and save_model:
                    current_timestamp_ms = datetime.now().strftime("%d_%m__%H_%M_%S")

                    self.model.save(f"models/model-M-{episode+1}-{current_timestamp_ms}.tf", save_format="tf")  # zapisz model w formacie h5
                # if (episode+1)%self.SAVE_MODEL_EVERY==0 and save_state:                 # zapisz bieżący stan uczenia
                #     pickle.dump((episode,episode_rewards,epsilon,self.replay_memory),open(f'models/{self.xid()}.pkl','wb'))
            to_restart=set()
            controls={}                                                                 # sterowania poszczególnych agentów
            for tname in self.env.agents:                                               # poruszamy każdym agentem
                if np.random.random()>epsilon:                                          # sterowanie wg reguły albo losowe
                    controls[tname]=np.argmax(self.decision(self.model,last_states[tname],current_states[tname]))
                    print('o',end='')
                else:
                    controls[tname]=np.random.randint(0,self.CTL_DIM)                   # losowa prędkość pocz. i skręt
                    print('.', end='')
            actions={tname:self.ctl2act(control) for tname,control in controls.items()} # wartości sterowań
            scene=self.env.step(actions)                                                # kroki i wyniki symulacji
            for tname,(new_state,reward,done) in scene.items():                         # obsługa po kroku dla każdego agenta
                episode_rewards[agent_episode[tname]]+=reward                           # akumulacja nagrody
                self.replay_memory.append((last_states[tname],current_states[tname],controls[tname],reward,new_state,done))
                step_cnt+=1
                if len(self.replay_memory)>=self.REPLAY_MEM_SIZE_MIN and step_cnt%self.TRAIN_EVERY==0:
                    self.do_train()                                      # ucz, gdy zgromadzono dość próbek
                    train_cnt+=1
                    if train_cnt%self.UPDATE_TARGET_EVERY==0:
                        self.target_model.set_weights(self.model.get_weights())         # aktualizuj model pomocniczy
                        print('T',end='')
                    else:
                        print('t',end='')
                if done:
                    to_restart.add(tname)
                    # print(f'\n {len(self.replay_memory)} {tname} E{episode} ',end='')
                    # print(f'{np.nanmean(episode_rewards.take(range(episode-self.env.MAX_STEPS-1,episode+1),mode="wrap"))/self.env.MAX_STEPS:.2f} ',end='')  # śr. nagroda za krok
                last_states[tname] = current_states[tname]                              # przejście do nowego stanu
                current_states[tname] = new_state                                       # z zapamiętaniem poprzedniego
                if epsilon > self.EPS_MIN:                                              # rosnące p-stwo uczenia na podst. historii
                    epsilon*=self.EPS_DECAY
                    epsilon=max(self.EPS_MIN,epsilon)                                   # ogranicz malenie eps

