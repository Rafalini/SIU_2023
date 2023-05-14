# encoding: utf8
import random
from datetime import datetime
import numpy as np
from time import time
from collections import deque
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv3D, Permute, Dense, Flatten
from .turtlesim_env_base import TurtlesimEnvBase
from .turtlesim_env_single import TurtlesimEnvSingle

class DqnSingle:
    # inicjalizacja parametrami domyślnymi, przechowanie dostarczonej referencji na środowisko symulacyjne
    def __init__(self,env:TurtlesimEnvBase,id_prefix='dqns',seed=42):
        self.env=env
        self.id_prefix=id_prefix            #   przyrostek identyfikatora modelu
        self.DISCOUNT=.9                    # D dyskonto dla nagrody w następnym kroku
        self.EPS_INIT=1.0                   #*  ε początkowy
        self.EPS_DECAY=.99                  #*E spadek ε
        self.EPS_MIN=.05                    #*e ε minimalny
        self.REPLAY_MEM_SIZE_MAX=20_000     # M rozmiar cache decyzji
        self.REPLAY_MEM_SIZE_MIN=4_000      # m zapełnienie warunkujące uczenie (4_000)
        self.MINIBATCH_SIZE=32              # B liczba decyzji w próbce uczącej
        self.TRAINING_BATCH_SIZE=self.MINIBATCH_SIZE // 4
        self.UPDATE_TARGET_EVERY=20         # U co ile treningów aktualizować model wolnozmienny
        self.EPISODES_MAX=4000              #*P liczba epizodów uczących
        self.CTL_DIM=6                      #   liczba możliwych akcji (tj. sterowań, decyzji)
        self.TRAIN_EVERY=4                  # T co ile kroków uczenie modelu szybkozmiennego
        self.SAVE_MODEL_EVERY=50            #*  co ile epizodów zapisywać model # TODO STUDENCI
        random.seed(seed)
        np.random.seed(seed)
        self.model=None
        self.target_model=None
        self.replay_memory=None
    # sygnatura eksperymentu, tj. wartości parametrów w jednym łańcych znaków - używane do nazywania plików z wynikami
    def xid(s) -> str:                      # 2 litery - parametr środowiska, 1 litera - parametr klasy uczącej
        return f'{s.id_prefix}-Gr{s.env.GRID_RES}_Cr{s.env.CAM_RES}_Sw{s.env.SPEED_RWRD_RATE}' \
               f'_Sv{s.env.SPEED_RVRS_RATE}_Sf{s.env.SPEED_FINE_RATE}_Dr{s.env.DIST_RWRD_RATE}' \
               f'_Oo{s.env.OUT_OF_TRACK_FINE}_Cd{s.env.COLLISION_DIST}_Ms{s.env.MAX_STEPS}' \
               f'_Pb{s.env.PI_BY}_D{s.DISCOUNT}_E{s.EPS_DECAY}_e{s.EPS_MIN}_M{s.REPLAY_MEM_SIZE_MAX}' \
               f'_m{s.REPLAY_MEM_SIZE_MIN}_B{s.MINIBATCH_SIZE}_U{s.UPDATE_TARGET_EVERY}' \
               f'_P{s.EPISODES_MAX}_T{s.TRAIN_EVERY}'
    # zakodowanie wybranego sterowania (0-5) na potrzeby środowiska: (prędkość,skręt)
    def ctl2act(_,decision:int):            # prędkość\skręt    -.1rad 0 .1rad
        v = .2                              #   0.2                0   1   2
        if decision >= 3:                   #   0.4                3   4   5
            v = .4
        w=.25*(decision%3-1)
        return [v,w]
    # złożenie dwóch rastrów sytuacji aktualnej i poprzedniej w tensor 5x5x8 wejścia do sieci
    def inp_stack(_,last,cur):
        # fa,fd,fc+1,fp+1 - z wyjścia get_map - BEZ 2 POCZ. WARTOŚCI (zalecana prędkość w ukł. odniesienia planszy)
        inp = np.stack([cur[2],cur[3],cur[4],cur[5],last[2],last[3],last[4],last[5]], axis=-1)
        return inp
    # predykcja nagród łącznych (Q) za sterowania na podst. bieżącej i ostatniej sytuacji
    def decision(self,the_model,last,cur):
        inp=np.expand_dims(self.inp_stack(last,cur),axis=-1)
        inp=np.expand_dims(inp,axis=0)
        return the_model(inp).numpy().flatten()             # wektor przewidywanych nagród dla sterowań
    # wytworzenie modelu - sieci neuronowej
    def make_model(self):
        N=self.env.GRID_RES                                                         # rozdzielczość rastra
        M=8                                                                         # liczba warstw z inp_stack()
        self.model=Sequential()
        self.model.add(Conv3D(filters=2*M,kernel_size=(2,2,M),activation='relu',input_shape=(N,N,M,1)))
        self.model.add(Permute((1,2,4,3)))
        self.model.add(Conv3D(filters=2*M,kernel_size=(2,2,2*M),activation='relu'))
        self.model.add(Permute((1,2,4,3)))
        self.model.add(Conv3D(filters=2*M,kernel_size=(2,2,2*M),activation='relu'))
        self.model.add(Flatten())
        self.model.add(Dense(32,activation='relu'))
        self.model.add(Dense(self.CTL_DIM,activation="linear"))                     # wyjście Q dla każdej z CTL_DIM decyzji
        self.model.compile(loss='mse',optimizer=keras.optimizers.Adam(learning_rate=0.001),metrics=["accuracy"])
    # uczenie od podstaw: generuj kroki, gromadź pomiary, ucz na próbce losowej, okresowo aktualizuj model pomocniczy

    def load_model(self, path):
        self.model = keras.models.load_model(path)

    def train_main(self,tname:str,save_model=True):             # TODO STUDENCI okresowy zapis modelu
        self.target_model=keras.models.clone_model(self.model)                      # model pomocniczy (wolnozmienny)
        self.replay_memory=deque(maxlen=self.REPLAY_MEM_SIZE_MAX)                   # historia kroków
        episode_rewards=np.zeros(self.EPISODES_MAX)*np.nan                          # historia nagród w epizodach
        epsilon=self.EPS_INIT
        step_cnt=0
        train_cnt=0
        for episode in range(self.EPISODES_MAX):                                    # ucz w epizodach treningowych
            print(f'{len(self.replay_memory)} E{episode} ',end='')
            current_state=self.env.reset(tnames=[tname],sections=['random'])[tname].map
            last_state=[i.copy() for i in current_state]                            # zaczyna od postoju: poprz. stan taki jak obecny
            episode_rwrd=0                                                          # suma nagród za kroki w epizodzie

            if save_model and episode%self.SAVE_MODEL_EVERY==0:                     # zapisuj co 250 epizodów gdy jest ustawiona flaga
                # current_timestamp_ms = round(time() * 1000)
                current_timestamp_ms = datetime.now().strftime("%d_%m__%H:%M:%S")
                self.model.save(f"models/model-E{episode}-{current_timestamp_ms}.tf", save_format="tf")  # zapisz model w formacie h5
                with open(f"models/model-E{episode}-{current_timestamp_ms}.config", "w+") as config_file:
                    config_file.write(self.xid())

            while True:                                                             # o przerwaniu decyduje do_train()
                if np.random.random()>epsilon:                                      # sterowanie wg reguły albo losowe
                    control=np.argmax(self.decision(self.model,last_state,current_state))
                    print('o',end='')                                               # "o" - sterowanie z modelu
                else:
                    control=np.random.randint(0,self.CTL_DIM)                       # losowa prędkość pocz. i skręt
                    print('.', end='')                                              # "." - sterowanie losowe
                new_state,reward,done=self.env.step({tname:self.ctl2act(control)})  #krok symulacji
                step_cnt+=1
                episode_rwrd+=reward
                self.replay_memory.append((last_state,current_state,control,reward,new_state,done))
                # bufor ruchów dość duży oraz przyszła pora by podtrenować model
                if len(self.replay_memory)>=self.REPLAY_MEM_SIZE_MIN and step_cnt%self.TRAIN_EVERY==0:
                    self.do_train()                              # ucz, gdy zgromadzono dość próbek
                    train_cnt+=1
                    if train_cnt%self.UPDATE_TARGET_EVERY==0:
                        self.target_model.set_weights(self.model.get_weights())     # aktualizuj model pomocniczy
                        print('T',end='')
                    else:
                        print('t',end='')
                if done:
                    break
                last_state = current_state                                          # przejście do nowego stanu
                current_state = new_state                                           # z zapamiętaniem poprzedniego
                if epsilon > self.EPS_MIN:                                          # rosnące p-stwo uczenia na podst. historii
                    epsilon*=self.EPS_DECAY
                    epsilon=max(self.EPS_MIN,epsilon)                               # podtrzymaj losowość ruchów
            episode_rewards[episode]=episode_rwrd
            print(f' Avg Reward: {np.nanmean(episode_rewards[episode-19:episode+1])/20:.2f}')   # śr. nagroda za krok

    # przygotowuje próbkę uczącą i wywołuje douczanie modelu
    def do_train(self):
        minibatch=random.sample(self.replay_memory,self.MINIBATCH_SIZE)             # losowy podzbiór kroków z historii
        Q0=np.zeros(((self.MINIBATCH_SIZE,self.CTL_DIM)))                           # nagrody krok n wg modelu bieżącego
        Q1target=Q0.copy()                                                          # nagrody krok n+1 wg modelu pomocniczego
        for idx,(last_state,current_state,_,_,new_state,_) in enumerate(minibatch):
            Q0[idx]=self.decision(self.model,last_state,current_state)              # krok n / model bieżący
            Q1target[idx]=self.decision(self.target_model,current_state,new_state)  # krok n+1 / model główny
        X = []  # sytuacje treningowe
        y = []  # decyzje treningowe
        for idx,(last_state,current_state,control,reward,new_state,done) in enumerate(minibatch):
            if done:
                new_q = reward                              # nie ma już stanu następnego, ucz się otrzymywać faktyczną nagrodę
            else:
                # nagroda uwzględnia nagrodę za kolejny etap sterowania
                new_q=reward + self.DISCOUNT*np.max(Q1target[idx])
            q0 = Q0[idx].copy()
            q0[control] = new_q                             # pożądane wyjście wg informacji po kroku (reszta - oszacowanie)
            inp = self.inp_stack(last_state,current_state)  # na wejściu - stan
            X.append(np.expand_dims(inp,axis=-1))
            y.append(q0)
        # douczanie modelu (w niektórych implementacjach wywoływana bezpośrednio propagacja wsteczna gradientu)
        # zapamiętanie wag tylko 1. warstwy splotowej
        X=np.stack(X)
        y=np.stack(y)
        self.model.fit(X,y,batch_size=self.TRAINING_BATCH_SIZE,verbose=0,shuffle=False)
