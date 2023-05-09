# encoding: utf8
import abc
import sys
import signal
import rospy
import turtlesim
import numpy as np
import random
import cv2 # noqa
from cv_bridge import CvBridge
from turtlesim.msg import Pose
from TurtlesimSIU import TurtlesimSIU

from src.utils import ScenarioReader


class TurtleAgent:      # struktura ze stałymi i bieżącymi atrybutami agenta
    pass                # route, sec_id, seq, color_api, goal_loc, pose, map, fd, section  - przypisywane na bieżąco


class TurtlesimEnvBase(metaclass=abc.ABCMeta):
    # określa parametry symulacji (poniższy zestaw i wartości sugerowane, nieobowiązkowe)
    def __init__(self):
        # parametry czujnika wizyjnego i interakcji z symulatorem
        self.GRID_RES = 5               # liczba komórek siatki
        self.CAM_RES = 200              # dł. boku siatki [px]
        # *okres dyskretyzacji sterowania - nie mniej niż 1 [s]
        self.SEC_PER_STEP = 1.0
        # oczekiwanie po setPose() i przed color_api.check() [s] (0.005 też daje radę)
        self.WAIT_AFTER_MOVE = .01
        # parametry oceny sytuacyjnej
        self.SPEED_RWRD_RATE = 0.5  # >wzmocnienie nagrody za jazdę w kierunku
        self.SPEED_RVRS_RATE = -10.0  # <wzmocnienie kary za jazdę pod prąd
        self.SPEED_FINE_RATE = -10.0  # <wzmocnienie kary za przekroczenie prędkości
        self.DIST_RWRD_RATE = 2.0  # >wzmocnienie nagrody za zbliżanie się do celu
        self.OUT_OF_TRACK_FINE = -10  # <ryczałtowa kara za wypadnięcie z trasy
        self.COLLISION_DIST = 1.5  # *odległość wykrycia kolizji [m]
        self.DETECT_COLLISION = False   # tryb wykrywania kolizji przez środowisko
        self.MAX_STEPS = 20             # maksymalna liczba kroków agentów
        self.PI_BY = 6  # *dzielnik zakresu pocz. odchylenia od azymutu żółwia na cel
        # aktualny stan środowiska symulacyjnego
        self.tapi = None                  # obiekt reprezentujący API symulatora
        self.px_meter_ratio = None        # skala planszy odczytana z symulatora
        # trasy agentów {id_trasy:(liczba_agentów,xmin,xmax,ymin,ymax,xg,yg)}
        self.routes = None
        # słownik stanów agentów {tname:TurtleAgent}
        self.agents = None
        self.step_sum = 0               # aktualna łączna liczba kroków wykonanych przez agenty
    # nawiązuje połączenie z symulatorem, ładuje trasy agentów i tworzy agenty (wywoływana jednokrotnie)

    def setup(self,
              routes_fname: str,        # nazwa pliku z definicją scenariuszy
              agent_cnt=None):          # ograniczenie na liczbę tworzonych agentów (None - brak ogr., agenty wg scenariuszy)
        # zainstalowanie obsługi zdarzenia
        signal.signal(signal.SIGINT, self.signal_handler)
        bridge = CvBridge()
        rospy.init_node('siu_example', anonymous=False)
        # ustanowienie komunikacji z symulatorem
        self.tapi = TurtlesimSIU.TurtlesimSIU()
        self.rate = rospy.Rate(1)
        self.px_meter_ratio = self.tapi.pixelsToScale()       # skala w pikselach na metr
        self.routes = {}
        self.agents = {}

        # TODO STUDENCI załadowanie tras agentów do self.routes     DONE!
        self.routes = ScenarioReader(scenario_file=routes_fname,
                                     px_meter_ratio=self.px_meter_ratio
                                    ).to_meters().get_routes()

        # utworzenie agentów-żółwi skojarzonych z trasami
        cnt = 0
        for route, sections in self.routes.items():          # dla kolejnych tras
            # dla kolejnych odcinków trasy
            for sec_id, sec in enumerate(sections):
                # utwórz określoną liczbę żółwi
                for seq in range(sec[0]):
                    cnt += 1
                    if agent_cnt is not None and cnt > agent_cnt:                 # ogranicz liczbę tworzonych żółwi
                        return
                    # identyfikator agenta: trasa, segment pocz., nr kolejny
                    tname = f'T_{route}_{sec_id}_{seq}'
                    # utwórz agenta lokalnie i zainicjuj tożsamość
                    ta = TurtleAgent()
                    ta.route = route
                    ta.sec_id = sec_id
                    ta.seq = seq
                    self.agents[tname] = ta
                    # utwórz/odtwórz agenta w symulatorze
                    if self.tapi.hasTurtle(tname):
                        self.tapi.killTurtle(tname)
                    self.tapi.spawnTurtle(tname, Pose())
                    self.tapi.setPen(
                        tname, turtlesim.srv.SetPenRequest(off=1))  # unieś rysik
                    # przechowuj obiekt sensora koloru
                    ta.color_api = TurtlesimSIU.ColorSensor(tname)
    # pozycjonuje żółwie na ich trasach , zeruje licznik kroków

    def reset(self,
              # lista nazw zółwi do resetu (None=wszystkie)
              tnames=None,
              sections='default') -> dict:                  # nry sekcji trasy dla każdego żółwia (default=domyślny,random=losowy)
        self.step_sum = 0
        if tnames is None:
            tnames = self.agents.keys()
            # powielenie sposobu pozycjonowania, jeśli chodzi o wszystkie żółwie
            sections = [sections for i in tnames]
        for tidx, tname in enumerate(tnames):
            agent = self.agents[tname]
            if sections[tidx] == 'default':                   # żółw pozycjonowany wg csv
                sec_id = agent.sec_id

            # żółw pozycjonowany w losowym segmencie jego trasy
            elif sections[tidx] == 'random':
                # TODO STUDENCI - chyba Done
                # losowanie obszaru proporcjonalnie do liczby planowanych żółwi w obszarze
                turtle_number = len(tnames)  # ilość żółwi
                # maksymalna liczba żółwi w jednym segmencie (można zmieniać w zależności od ilości żółwi)
                MAX_NUMBER_IN_SEC = 3
                max_numb = MAX_NUMBER_IN_SEC
                if turtle_number > MAX_NUMBER_IN_SEC * 8:  # ilość segmentów na maksymalna ilość żółwi
                    max_numb = turtle_number/8 + 1
                # losowanie liczby od 0 do 7 (indeks obszaru)
                rand_sec = random.randint(0, 7)

                # Sprawdzenie liczby żółwi w wylosowanym obszarze
                if tidx != 0:
                    turtle_count = sum([1 for t in list(self.agents.values())[
                                       :tidx] if t.section == rand_sec])
                    # Jeśli liczba żółwi w obszarze jest większa lub równa maksymalnej liczbie,
                    # losuj kolejny obszar aż do znalezienia obszaru z mniejszą liczbą żółwi
                    while turtle_count >= max_numb:
                        rand_sec = random.randint(0, 7)
                        turtle_count = sum([1 for t in list(self.agents.values())[
                                           :tidx] if t.section == sec_id])

                # Przypisanie wylosowanego obszaru jako sec_id
                sec_id = rand_sec
            # żółw pozycjonowany we wskazanym segmencie (liczone od 0)
            else:
                sec_id = sections[tidx]
            # przypisanie sekcji, w której się odrodzi
            section = self.routes[agent.route][sec_id]
            agent.goal_loc = Pose(x=section[5], y=section[6])  # pierwszy cel
            # próba ulokowania agenta we wskazanym obszarze i jednocześnie na drodze (niezerowy wektor zalecanej prędkości)
            while True:
                x = np.random.uniform(section[1], section[2])
                y = np.random.uniform(section[3], section[4])
                # azymut początkowy dokładnie w kierunku celu
                theta = np.arctan2(agent.goal_loc.y-y, agent.goal_loc.x-x)
                # przestawienie żółwia w losowe miejsce obszaru narodzin
                self.tapi.setPose(tname, Pose(
                    x=x, y=y, theta=theta), mode='absolute')
                # odczekać UWAGA inaczej symulator nie zdąży przestawić żółwia
                rospy.sleep(self.WAIT_AFTER_MOVE)
                fx, fy, _, _, _, _ = self.get_road(
                    tname)            # fx, fy \in <-1,1>
                fo = self.get_map(tname)[6]
                if self.DETECT_COLLISION and fo[self.GRID_RES//2, self.GRID_RES-1] == 0:
                    # wykrywanie kolizji na początku (GRID_RES-1) środkowego wiersza (GRID_RES//2) rastra
                    continue
                if abs(fx)+abs(fy) > .01:                         # w obrębie drogi
                    # kierunek ruchu zaburzony +- pi/PI_BY rad.
                    theta += np.random.uniform(-np.pi /
                                               self.PI_BY, np.pi/self.PI_BY)
                    # obrót żółwia
                    p = Pose(x=x, y=y, theta=theta)
                    self.tapi.setPose(tname, p, mode='absolute')
                    agent.pose = p                                # zapamiętanie azymutu, lokalnie
                    rospy.sleep(self.WAIT_AFTER_MOVE)
                    break                                       # udana lokalizacja, koniec prób
            # zapamiętanie otoczenia, lokalnie
            agent.map = self.get_map(tname)
            # zapamiętanie sekcji, w której się odradza
            agent.section = sec_id
        return self.agents
    # wykonuje zlecone działania, zwraca sytuacje żółwi, nagrody, flagi końca przejazdu

    @abc.abstractmethod
    def step(self,
             # działania żółwi, actions={tname:(speed,turn)}
             actions: dict,
             realtime=False) -> dict:                          # flaga sterowania prędkościowego
        pass
    # zainstalowana procedura obsługi przerwania

    def signal_handler(self, sig, frame):
        print("Terminating")
        sys.exit(0)

    def get_road(self, tname):
        agent = self.agents[tname]
        print(tname, agent.color_api)
        # bez tego color_api.check() nie wyrabia
        rospy.sleep(self.WAIT_AFTER_MOVE)
        # kolor planszy pod żółwiem
        color = agent.color_api.check()
        # składowa x zalecanej prędkości <-1;1>
        fx = .02*(color.r-200)
        # składowa y zalecanej prędkości <-1;1>
        fy = .02*(color.b-200)
        # mnożnik kary za naruszenie ograniczeń prędkości
        fa = color.g/255.0
        # aktualna pozycja żółwia
        pose = self.tapi.getPose(tname)
        fd = np.sqrt((agent.goal_loc.x-pose.x)**2 +
                     (agent.goal_loc.y-pose.y)**2)     # odl. do celu
        # rzut zalecanej prędkości na azymut
        fc = fx*np.cos(pose.theta)+fy*np.sin(pose.theta)
        # rzut zalecanej prędkości na _|_ azymut
        fp = fy*np.cos(pose.theta)-fx*np.sin(pose.theta)
        return fx, fy, fa, fd, fc+1, fp+1
    # zwraca macierze opisujące sytuację w otoczeniu wskazanego agenta (tname)

    def get_map(self, tname: str):
        agent = self.agents[tname]
        pose = self.tapi.getPose(tname)
        # pobranie rastra sytuacji żółwia - każda komórka zawiera uśr. kolor planszy, odl. od celu i flagę obecności innych agentów
        img = self.tapi.readCamera(tname,
                                   frame_pixel_size=self.CAM_RES,
                                   cell_count=self.GRID_RES**2,
                                   x_offset=0,
                                   goal=agent.goal_loc,
                                   show_matrix_cells_and_goal=False)
        # przygotowanie macierzy z informacją sytuacyjną - kanał czerwony
        fx = np.eye(self.GRID_RES)
        fy = fx.copy()                                            # kanał niebieski
        fa = fx.copy()                                            # kanał zielony
        fd = fx.copy()                                            # odl. od celu
        # obecność innego agenta w komórce (occupied)
        fo = fx.copy()
        for i, row in enumerate(img.m_rows):
            for j, cell in enumerate(row.cells):
                fx[i, j] = cell.red                              # <-1,1>
                fy[i, j] = cell.blue
                fa[i, j] = cell.green
                fd[i, j] = cell.distance                         # [m]
                # {0,1}^GRID_RES^2
                fo[i, j] = cell.occupy
        # rzut zalecanej prędkości na azymut [m/s]
        fc = fx*np.cos(pose.theta)+fy*np.sin(pose.theta)
        # rzut zalecanej prędkości na _|_ azymut
        fp = fy*np.cos(pose.theta)-fx*np.sin(pose.theta)
        # informacja w ukł. wsp. żółwia
        return fx, fy, fa, fd, fc+1, fp+1, fo
