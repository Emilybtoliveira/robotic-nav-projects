from GUI import GUI
from HAL import HAL
from MAP import MAP
import numpy as np
import math
from queue import PriorityQueue
from queue import Queue
import time

def drive(grid):
  while True:
    current_pose = HAL.getPose3d()
    x, y = MAP.rowColumn((current_pose.x, current_pose.y))

    if grid[y][x] == 0:
        HAL.setV(0)
        HAL.setW(0)
        return

    lowest_val = grid[y][x]
    lowest_coord = (0, 0)

    for i in range(-11, 12, 1):
        for j in range(-11, 12, 1):
            try:
                if map[y + j][x + i] != 0 and grid[y + j][x + i] >= 0 and grid[y + j][x + i] < lowest_val:
                    lowest_val = grid[y + j][x + i]
                    lowest_coord = (i, j)
            except:
                pass

    theta = math.atan2(-lowest_coord[1], -lowest_coord[0])
    angle_diff = current_pose.yaw - theta

    if angle_diff > math.pi:
        angle_diff -= 2 * math.pi
    elif angle_diff < -math.pi:
        angle_diff += 2 * math.pi

    HAL.setW(-angle_diff)

    linear_speed = math.sqrt(lowest_coord[0] ** 2 + lowest_coord[1] ** 2) * 0.5
    linear_speed = max(linear_speed, 2)  
    HAL.setV(linear_speed)

def createGrid():
  return [[-1 for _ in range(400)] for _ in range(400)]

def calculateWeights(queue, item, grid):
  x, y = item[1]
  value = item[0]

  for i, j in [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
      if 0 <= y + j < 400 and 0 <= x + i < 400 and grid[y + j][x + i] == -1 and map[y + j][x + i] != 0:
          queue.put((value + math.sqrt(i ** 2 + j ** 2), (x + i, y + j)))
          grid[y + j][x + i] = value + math.sqrt(i ** 2 + j ** 2)

  return queue

def addBorder():
  border_grid = [[0 for _ in range(400)] for _ in range(400)]

  for x in range(400):
      for y in range(400):
          if map[y][x] == 0:
              for i in range(-5, 6):
                  for j in range(-5, 6):
                      try:
                          if map[y + j][x + i] != 0:
                              border_grid[y + j][x + i] += 5 - abs(j) + 5 - abs(i)
                      except:
                          pass

  return border_grid

def normalize(grid):
  return np.clip(grid, 0, 255).astype('uint8')

map = MAP.getMap('/RoboticsAcademy/exercises/static/exercises/global_navigation_newmanager/resources/images/cityLargenBin.png')

while True:
    target = None

    while target is None:
        print("Waiting for target...")
        target = GUI.getTargetPose()

    target_coord = MAP.rowColumn(target)
    print(f"Heading to {target_coord}")

    car_coord = MAP.rowColumn((HAL.getPose3d().x, HAL.getPose3d().y))

    grid = createGrid()
    border_grid = addBorder()
    q = PriorityQueue()
    q.put((0, target_coord))
    grid[target_coord[1]][target_coord[0]] = 0
    extra_steps = 10000
    start_extra = False

    print_each = 0

    while not q.empty() and extra_steps > 0:
        item = q.get()
        q = calculateWeights(q, item, grid)

        if start_extra:
            extra_steps -= 1

        if item[1][0] == car_coord[0] and item[1][1] == car_coord[1]:
            start_extra = True

        print_each += 1

        if print_each > 250:
            print_each = 0
            GUI.showNumpy(normalize(grid))

    for y in range(len(grid)):
        for x in range(len(grid[0])):
            if grid[y][x] == -1:
                grid[y][x] = 0
            else:
                grid[y][x] += border_grid[y][x]

    grid[target_coord[1]][target_coord[0]] = 0
    grid_normalized = normalize(grid)
    GUI.showNumpy(grid_normalized)

    drive(grid)
    print("Destination reached")