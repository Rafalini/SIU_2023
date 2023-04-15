# encoding: utf8
import abc
import sys
import csv
import signal
import rospy
import turtlesim
import numpy as np
from cv_bridge import CvBridge
from turtlesim.msg import Pose
from TurtlesimSIU import TurtlesimSIU

class TurtleAgent:      # struktura ze stałymi i bieżącymi atrybutami agenta
    pass                # route, sec_id, seq, color_api, goal_loc, pose, map, fd, section  - przypisywane na bieżąco

class TurtlesimEnvBase(metaclass=abc.ABCMeta):
    # określa parametry symulacji (poniższy zestaw i wartości sugerowane, nieobowiązkowe)
    def __init__(self):
        # parametry czujnika wizyjnego i interakcji z symulatorem
        self.GRID_RES = 5               # liczba komórek siatki
        self.CAM_RES = 200              # dł. boku siatki [px]
        self.SEC_PER_STEP = 1.0         #*okres dyskretyzacji sterowania - nie mniej niż 1 [s]
        self.WAIT_AFTER_MOVE = .01      # oczekiwanie po setPose() i przed color_api.check() [s] (0.005 też daje radę)
        # parametry oceny sytuacyjnej
        self.SPEED_RWRD_RATE = 0.5      #>wzmocnienie nagrody za jazdę w kierunku
        self.SPEED_RVRS_RATE = -10.0    #<wzmocnienie kary za jazdę pod prąd
        self.SPEED_FINE_RATE = -10.0    #<wzmocnienie kary za przekroczenie prędkości
        self.DIST_RWRD_RATE = 2.0       #>wzmocnienie nagrody za zbliżanie się do celu
        self.OUT_OF_TRACK_FINE = -10    #<ryczałtowa kara za wypadnięcie z trasy
        self.COLLISION_DIST = 1.5       #*odległość wykrycia kolizji [m]
        self.DETECT_COLLISION = False   # tryb wykrywania kolizji przez środowisko
        self.MAX_STEPS = 20             # maksymalna liczba kroków agentów
        self.PI_BY = 6                  #*dzielnik zakresu pocz. odchylenia od azymutu żółwia na cel
        # aktualny stan środowiska symulacyjnego
        self.tapi=None                  # obiekt reprezentujący API symulatora
        self.px_meter_ratio=None        # skala planszy odczytana z symulatora
        self.routes = None              # trasy agentów {id_trasy:(liczba_agentów,xmin,xmax,ymin,ymax,xg,yg)}
        self.agents = None              # słownik stanów agentów {tname:TurtleAgent}
        self.step_sum = 0               # aktualna łączna liczba kroków wykonanych przez agenty
    # nawiązuje połączenie z symulatorem, ładuje trasy agentów i tworzy agenty (wywoływana jednokrotnie)
    def setup(self,
              routes_fname: str,        # nazwa pliku z definicją scenariuszy
              agent_cnt=None):          # ograniczenie na liczbę tworzonych agentów (None - brak ogr., agenty wg scenariuszy)
        signal.signal(signal.SIGINT,self.signal_handler)    # zainstalowanie obsługi zdarzenia
        bridge = CvBridge()
        rospy.init_node('siu_example',anonymous=False)
        self.tapi=TurtlesimSIU.TurtlesimSIU()               # ustanowienie komunikacji z symulatorem
        self.rate=rospy.Rate(1)
        self.px_meter_ratio=self.tapi.pixelsToScale()       # skala w pikselach na metr
        self.routes={}
        self.agents={}

        # TODO STUDENCI załadowanie tras agentów do self.routes     DONE!
        with open(routes_fname,encoding='utf-8-sig') as f:  # załadowanie tras agentów

            csv_reader = csv.reader(f)
            sequences=[]

            for row in csv_reader:
                split_row=row[0].split(';')
                split_row_ints = list(map(int, split_row))      # Map items in list from 'str' to 'int' 
                route_id=split_row_ints[0]
                sequences.append(split_row_ints[1:])

            self.routes[route_id]=sequences
            f.close()

        # utworzenie agentów-żółwi skojarzonych z trasami
        cnt=0
        for route,sections in self.routes.items():          # dla kolejnych tras
            for sec_id,sec in enumerate(sections):          # dla kolejnych odcinków trasy
                for seq in range(sec[0]):                   # utwórz określoną liczbę żółwi
                    cnt +=1
                    if agent_cnt is not None and cnt>agent_cnt:                 # ogranicz liczbę tworzonych żółwi
                        return
                    tname=f'{route}_{sec_id}_{seq}'         # identyfikator agenta: trasa, segment pocz., nr kolejny
                    print(f'Agent {tname}')
                    ta=TurtleAgent()                        # utwórz agenta lokalnie i zainicjuj tożsamość
                    ta.route=route
                    ta.sec_id=sec_id
                    ta.seq=seq
                    self.agents[tname]=ta
                    if self.tapi.hasTurtle(tname):          # utwórz/odtwórz agenta w symulatorze
                        self.tapi.killTurtle(tname)
                    self.tapi.spawnTurtle(tname,Pose())
                    self.tapi.setPen(tname,turtlesim.srv.SetPenRequest(off=1))  # unieś rysik
                    ta.color_api=TurtlesimSIU.ColorSensor(tname)                # przechowuj obiekt sensora koloru
    # pozycjonuje żółwie na ich trasach , zeruje licznik kroków
    def reset(self,
              tnames=None,                                  # lista nazw zółwi do resetu (None=wszystkie)
              sections='default') -> dict:                  # nry sekcji trasy dla każdego żółwia (default=domyślny,random=losowy)
        self.step_sum = 0
        if tnames is None:
            tnames = self.agents.keys()
            sections=[sections for i in tnames]             # powielenie sposobu pozycjonowania, jeśli chodzi o wszystkie żółwie
        for tidx,tname in enumerate(tnames):
            agent = self.agents[tname]
            if sections[tidx]=='default':                   # żółw pozycjonowany wg csv
                sec_id=agent.sec_id
            elif sections[tidx]=='random':                  # żółw pozycjonowany w losowym segmencie jego trasy
                # TODO STUDENCI
                # losowanie obszaru proporcjonalnie do liczby planowanych żółwi w obszarze
                sec_id=...
            else:                                           # żółw pozycjonowany we wskazanym segmencie (liczone od 0)
                sec_id=sections[tidx]
            section=self.routes[agent.route][sec_id]        # przypisanie sekcji, w której się odrodzi
            agent.goal_loc=Pose(x=section[5],y=section[6])  # pierwszy cel
            # próba ulokowania agenta we wskazanym obszarze i jednocześnie na drodze (niezerowy wektor zalecanej prędkości)
            while True:
                x = np.random.uniform(section[1],section[2])
                y = np.random.uniform(section[3],section[4])
                # azymut początkowy dokładnie w kierunku celu
                theta = np.arctan2(agent.goal_loc.y-y,agent.goal_loc.x-x)
                # przestawienie żółwia w losowe miejsce obszaru narodzin
                self.tapi.setPose(tname,Pose(x=x,y=y,theta=theta),mode='absolute')
                rospy.sleep(self.WAIT_AFTER_MOVE)               # odczekać UWAGA inaczej symulator nie zdąży przestawić żółwia
                fx,fy,_,_,_,_ = self.get_road(tname)            # fx, fy \in <-1,1>
                fo = self.get_map(tname)[6]
                if self.DETECT_COLLISION and fo[self.GRID_RES//2,self.GRID_RES-1] == 0:
                    continue             # wykrywanie kolizji na początku (GRID_RES-1) środkowego wiersza (GRID_RES//2) rastra
                if abs(fx)+abs(fy)>.01:                         # w obrębie drogi
                    theta+=np.random.uniform(-np.pi/self.PI_BY,np.pi/self.PI_BY)    # kierunek ruchu zaburzony +- pi/PI_BY rad.
                    p = Pose(x=x,y=y,theta=theta)               # obrót żółwia
                    self.tapi.setPose(tname,p,mode='absolute')
                    agent.pose=p                                # zapamiętanie azymutu, lokalnie
                    rospy.sleep(self.WAIT_AFTER_MOVE)
                    break                                       # udana lokalizacja, koniec prób
            agent.map = self.get_map(tname)                     # zapamiętanie otoczenia, lokalnie
            agent.section=sec_id                                # zapamiętanie sekcji, w której się odradza
        return self.agents
    # wykonuje zlecone działania, zwraca sytuacje żółwi, nagrody, flagi końca przejazdu
    @abc.abstractmethod
    def step(self,
              actions: dict,                                    # działania żółwi, actions={tname:(speed,turn)}
              realtime=False) -> dict:                          # flaga sterowania prędkościowego
        pass
    # zainstalowana procedura obsługi przerwania
    def signal_handler(self,sig,frame):
        print("Terminating")
        sys.exit(0)
    def get_road(self,tname):
        agent = self.agents[tname]
        print(tname,agent.color_api)
        rospy.sleep(self.WAIT_AFTER_MOVE)                       # bez tego color_api.check() nie wyrabia
        color = agent.color_api.check()                         # kolor planszy pod żółwiem
        fx = .02*(color.r-200)                                  # składowa x zalecanej prędkości <-1;1>
        fy = .02*(color.b-200)                                  # składowa y zalecanej prędkości <-1;1>
        fa = color.g/255.0                                      # mnożnik kary za naruszenie ograniczeń prędkości
        pose = self.tapi.getPose(tname)                         # aktualna pozycja żółwia
        fd = np.sqrt((agent.goal_loc.x-pose.x)**2+(agent.goal_loc.y-pose.y)**2)     # odl. do celu
        fc = fx*np.cos(pose.theta)+fy*np.sin(pose.theta)        # rzut zalecanej prędkości na azymut
        fp = fy*np.cos(pose.theta)-fx*np.sin(pose.theta)        # rzut zalecanej prędkości na _|_ azymut
        return fx,fy,fa,fd,fc+1,fp+1
    # zwraca macierze opisujące sytuację w otoczeniu wskazanego agenta (tname)
    def get_map(self,tname: str):
        agent = self.agents[tname]
        pose  = self.tapi.getPose(tname)
        # pobranie rastra sytuacji żółwia - każda komórka zawiera uśr. kolor planszy, odl. od celu i flagę obecności innych agentów
        img   = self.tapi.readCamera(tname,
                                     frame_pixel_size=self.CAM_RES,
                                     cell_count=self.GRID_RES**2,
                                     x_offset=0,
                                     goal=agent.goal_loc,
                                     show_matrix_cells_and_goal=False)
        fx=np.eye(self.GRID_RES)                                # przygotowanie macierzy z informacją sytuacyjną - kanał czerwony
        fy=fx.copy()                                            # kanał niebieski
        fa=fx.copy()                                            # kanał zielony
        fd=fx.copy()                                            # odl. od celu
        fo=fx.copy()                                            # obecność innego agenta w komórce (occupied)
        for i,row in enumerate(img.m_rows):
            for j,cell in enumerate(row.cells):
                fx[i,j] = cell.red                              # <-1,1>
                fy[i,j] = cell.blue
                fa[i,j] = cell.green
                fd[i,j] = cell.distance                         # [m]
                fo[i,j] = cell.occupy                           # {0,1}^GRID_RES^2
        fc=fx*np.cos(pose.theta)+fy*np.sin(pose.theta)          # rzut zalecanej prędkości na azymut [m/s]
        fp=fy*np.cos(pose.theta)-fx*np.sin(pose.theta)          # rzut zalecanej prędkości na _|_ azymut
        return fx,fy,fa,fd,fc+1,fp+1,fo                         # informacja w ukł. wsp. żółwia
