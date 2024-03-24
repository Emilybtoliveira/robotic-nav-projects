from GUI import GUI
from HAL import HAL
from ompl import base as ob
from ompl import geometric as og
from math import sqrt
import math
import numpy as np
import time

IMAGE_H = 279
IMAGE_W = 415
SHELF_COORD  = [3.728, -1.242]
OBSTACLES_SIZE = 3
DISTANCE_ERR = 0.1

MOVE_TO = 0
SHELF_POS = 1
RETURN = 2
FINISHED = 3

obstacles = []
map_dims = [0, 0, IMAGE_H, IMAGE_W]
state = MOVE_TO
path_index = 1
error = 0

map = GUI.getMap('/RoboticsAcademy/exercises/static/exercises/amazon_warehouse_newmanager/resources/images/map.png')
velocity = 1.0


def isValid(state):

  x = round(state.getX())
  y = round(state.getY())

  if [x, y] not in obstacles:
    for i in range(len(obstacles)):
      if sqrt(pow(x - obstacles[i][0], 2) + pow(y - obstacles[i][1], 2)) - 4 <= 0:
        return False
    return True
  return False

def generateObstacles(world_map):
  for row in range(IMAGE_H):
    for col in range(IMAGE_W):
      if 0 < row < IMAGE_H and 0 < col < IMAGE_W and np.any(world_map[row, col, :3] == 0.):
        for r in range(row - OBSTACLES_SIZE, row + OBSTACLES_SIZE + 1): # With 8 works
          for c in range(col - OBSTACLES_SIZE, col + OBSTACLES_SIZE + 1):
            if [r, c] not in obstacles and 0 <= r < IMAGE_H and 0 <= c < IMAGE_W :
              print([r,c])
              obstacles.append([r, c])

def world_2_map(coordx, coordy):

  new_row = round(-20.4411764705882*coordx + 139)
  new_col = round(-20.0775945683802*coordy + 207)
  
  return new_row, new_col
  
def map_2_world(row, col):
  posx = -0.0489208633094*row + 6.8
  posy = -0.049806763285*col + 10.31
  
  return posx, posy

def plan(destx, desty):
  space = ob.SE2StateSpace()

  bounds = ob.RealVectorBounds(2)
  bounds.setLow(1, map_dims[1])
  bounds.setLow(0, map_dims[0])
  bounds.setHigh(0, map_dims[2])
  bounds.setHigh(1, map_dims[3])
  space.setBounds(bounds)

  si = ob.SpaceInformation(space)
  si.setStateValidityChecker(ob.StateValidityCheckerFn(isValid))

  start = ob.State(space)
  start_posx, start_posy = world_2_map(HAL.getPose3d().x, HAL.getPose3d().y)
  start().setX(start_posx)
  start().setY(start_posy)
  start().setYaw(HAL.getPose3d().yaw)
  goal = ob.State(space)
  goal().setX(destx)
  goal().setY(desty)
  goal().setYaw(0)

  pdef = ob.ProblemDefinition(si)

  pdef.setStartAndGoalStates(start, goal)

  planner = og.RRTConnect(si)
  planner.setRange(30)
  planner.setProblemDefinition(pdef)

  planner.setup()

  solved = planner.solve(20.0)
  if solved:
    path = create_numpy_path(pdef.getSolutionPath().printAsMatrix())
    GUI.showPath(path)
    return path
  else:
    print("NOT FOUND")
  
def create_numpy_path(states):
  lines = states.splitlines()
  length = len(lines) - 1
  array = np.zeros((length, 2))
  for i in range(length):
      array[i][1] = round(float(lines[i].split(" ")[0]))
      array[i][0] = round(float(lines[i].split(" ")[1]))
  return array

def angular_vel(setpoint, prev_error):
  Kp = 0.7
  Kd = 0.35
  
  setpoint_posx, setpoint_posy = map_2_world(setpoint[1], setpoint[0])
  set_angle = math.atan2(setpoint_posy - HAL.getPose3d().y, setpoint_posx - HAL.getPose3d().x)
  curr_erroror = set_angle - HAL.getPose3d().yaw
  w = Kp*curr_erroror + Kd*(curr_erroror - prev_error)
  return w, curr_erroror


generateObstacles(map)
print("Mapa gerado")
dest_x, dest_y = world_2_map(SHELF_COORD[0], SHELF_COORD[1])
path_arr = plan(dest_x, dest_y)
print(path_arr)

while True:
    if state == MOVE_TO:
      print("starting")
      HAL.setV(velocity)
      ang_v, error = angular_vel(path_arr[path_index], error)
      HAL.setW(ang_v)
      path_x, path_y = map_2_world(path_arr[path_index][1], path_arr[path_index][0])
      print(path_x, path_y)
      
      if (abs(path_x - HAL.getPose3d().x) <= DISTANCE_ERR and
        abs(path_y - HAL.getPose3d().y) <= DISTANCE_ERR):
          path_index += 1
          
      if path_index == len(path_arr):
        state = SHELF_POS
      
    elif state == SHELF_POS:
      print("got to the shelve")
      HAL.setV(0)
      
      if round(HAL.getPose3d().yaw - math.pi) != 0:
        HAL.setW(0.1)
        
      else:
        HAL.setW(0)
        time.sleep(1)
        HAL.lift()
        state = RETURN
        path_index = 0
        error = 0
        
    elif state == RETURN:   
      print("returning...")
      HAL.setV(velocity)
      ang_v, error = angular_vel(path_arr[path_index], error)
      HAL.setW(ang_v)
      path_x, path_y = map_2_world(path_arr[path_index][1], path_arr[path_index][0])
      
      if (abs(path_x - HAL.getPose3d().x) <= DISTANCE_ERR and
        abs(path_y - HAL.getPose3d().y) <= DISTANCE_ERR):
          path_index -= 1
          
      if path_index == -1:
        HAL.putdown()
        HAL.setV(0)
        HAL.setW(0)
        state = FINISHED
        print("finished!")
    