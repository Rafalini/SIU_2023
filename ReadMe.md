# Turtlesim simulation

Python based reinforced learning enviroment. Aim of this project was to implement reinforced learning alghoritms using python and test them in visual way. 

Turtlesim allows to graphically simulate turtle movement on route.

![image](https://github.com/Rafalini/SIU_2023/assets/44322872/476b87d4-0364-4e0a-a343-5f460e36f09e)


## Repository contents

| Folder        | Contents |
| -----------   | ----------- |
| Data          | Training scenarios in scv fiels       |
| models        | Pre-trained models in tensorflow format        |
| src           | Source files of turtlesim, visualisation and training module        |
| static        | Example maps in png format        |

And scripts to run training and test trained models: train.py and run.py for single turtle, trainMulti.py and runMulti.py for multiple turtles running simultaneously.

## Map encoding

Map is presented in png format, it is possible to draw custom map. Included models where trained to apporximate following movement schema:

Area excluded from turtles is marked on *Green* channel with value *255*. Directions of movement are encoded with Red and Blue channel:

Colors:
| |  | |
| -----------   | ----------- | ----------- |
|       | B 250 |       |
| R 150 |       | R 250 |
|       | B 150 |       |

Directions on map:
| |  | |
| -----------   | ----------- | ----------- |
|       | North |       |
| West |       | East |
|       | South |       |

## Scenarios

Scenarios consist of:

| episode_id | number_of_turtles | x_min | x_max | y_min | y_max | x_target | y_target |
| -----------   | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- |

where min and max coordinates mark starting area (rectangle - its bottom left corner and top right) and target coordinates mark target to achieve during training. Ilustration of training areas and targets is static foler. For training it is advised to place target area beyound maximal distance that can travel turtle (parameter MAX_STEPS in turtlesim_env_base.py) to achieve good training results, it prevents turtle from spining around target point and gaining reward for this actions - leraning bad habbit.

## Requirements

Docker is required to run code in container with integrated simulator.

## How to run

1. Start container & mount local directory (models, scenarios etc):
``` 
docker run --name siu -p 6080:80 -e RESOLUTION=1366x768 --mount type=bind,source="$(pwd)",target=/root/ shelvi96/siu:0.1.1 
```
container is visible by deafult at: ```localhost:6080```
2. Start console (system tools -> terminal) and install pip and tensorflow:
``` 
$ apt install pip
$ pip install tensorflow
```
3. Replace source map file with desired map, source map is located in file: ```/roads.png```
```
cp ./static/board_A.png /roads.png
```
4. Run simulation enviroment in one terminal:
```
roslaunch turtlesim siu.launch
```
4. Begin training or run trained model using provided scripts in next terminal:
[turtles.webm](https://github.com/Rafalini/SIU_2023/assets/44322872/423945af-7b83-4a87-9172-036e056fedb3)



