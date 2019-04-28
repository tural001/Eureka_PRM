import numpy as np
import pylab as pl
import sys
import random
import scipy.spatial
import math

sys.path.append('osr_examples/scripts/')

import environment_2d

pl.ion()
np.random.seed(4)

env = environment_2d.Environment(10, 6, 5)

pl.clf()
env.plot()

q = env.random_query()

if q is not None:
  x_start, y_start, x_goal, y_goal = q
  env.plot_query(x_start, y_start, x_goal, y_goal)




def random_points(x_start, y_start, x_goal, y_goal):

    # random_points function returns x, y coordinates of n points. Increase n for more efficient (better) solution. The probability of exact solution converges to 1 if n approaches infinity according to PRM method.
    maxx = 10
    maxy = 6
    minx = 0
    miny = 0

    random_x, random_y = [], []
    n=1000

    while len(random_x) <= n:
        tx = (random.random() - minx) * (maxx - minx)
        ty = (random.random() - miny) * (maxy - miny)

        if not env.check_collision(tx, ty):
            random_x.append(tx)
            random_y.append(ty)

    random_x.append(x_start)
    random_y.append(y_start)
    random_x.append(x_goal)
    random_y.append(y_goal)

    return random_x, random_y
												


class Node:

    # Node class for dijkstra search

    def __init__(self, x, y, cost, pind):
        self.x = x
        self.y = y
        self.cost = cost
        self.pind = pind

    def __str__(self):
        return str(self.x) + "," + str(self.y) + "," + str(self.cost) + "," + str(self.pind)

class KDTree:

    # Nearest neighbor search class with KDTree

    def __init__(self, data):
        # store kd-tree
        self.tree = scipy.spatial.cKDTree(data)

    def search(self, inp, k=1):
        """
        Search NN
        inp: input data, single frame or multi frame
        """

        if len(inp.shape) >= 2:  # multi input
            index = []
            dist = []

            for i in inp.T:
                idist, iindex = self.tree.query(i, k=k)
                index.append(iindex)
                dist.append(idist)

            return index, dist

        dist, index = self.tree.query(inp, k=k)
        return index, dist

    def search_in_distance(self, inp, r):
        """
        find points with in a distance r
        """

        index = self.tree.query_ball_point(inp, r)
        return index

def generate_roadmap(sample_x, sample_y):


    road_map = []
    nsample = len(random_x)
    skdtree = KDTree(np.vstack((random_x, random_y)).T)

    for (i, ix, iy) in zip(range(nsample), random_x, random_y):

        index, dists = skdtree.search(
            np.array([ix, iy]).reshape(2, 1), k=nsample)
        inds = index[0]
        edge_id = []

        for ii in range(1, len(inds)):
            nx = random_x[inds[ii]]
            ny = random_y[inds[ii]]

            d = math.sqrt((nx-ix)**2+(ny-iy)**2)
            if d > 2: #if length of each edge is longer than 2 then ignore
                break
            n=200  #number of equally seperated points in each line considered for collision check. If non of the points on the line coincide with obstacle points, then this line is appended to road_map
            dx=(nx-ix)/n
            dy=(ny-iy)/n
            points = []
            for k in range(n):
                points.append(((ix+k*dx), (iy+k*dy)))

            line_on_obstacles = False
            for j in range(n):
                x0, y0 = points[j]
                if env.check_collision(x0, y0):
                    line_on_obstacles = True
                    break
            if not line_on_obstacles:
                edge_id.append(inds[ii])
                pl.plot([ix, nx], [iy, ny], 'yo-')
            if len(edge_id) >= 10:
                break

        road_map.append(edge_id)

    return road_map



def dijkstra_planning(x_start, y_start, x_goal, y_goal, road_map, random_x, random_y):

    nstart = Node(x_start, y_start, 0.0, -1)
    ngoal = Node(x_goal, y_goal, 0.0, -1)

    openset, closedset = dict(), dict()
    openset[len(road_map) - 2] = nstart

    while True:
        if not openset:
            print("Cannot find path")
            break

        c_id = min(openset, key=lambda o: openset[o].cost)
        current = openset[c_id]

        # show graph
        if len(closedset.keys()) % 2 == 0:
            pl.plot(current.x, current.y, "xg")
            pl.pause(0.001)

        if c_id == (len(road_map) - 1):
            print("goal is found!")
            ngoal.pind = current.pind
            ngoal.cost = current.cost
            break

        # Remove the item from the open set
        del openset[c_id]
        # Add it to the closed set
        closedset[c_id] = current

        # expand search grid based on motion model
        for i in range(len(road_map[c_id])):
            n_id = road_map[c_id][i]
            dx = random_x[n_id] - current.x
            dy = random_y[n_id] - current.y
            d = math.sqrt(dx**2 + dy**2)
            node = Node(random_x[n_id], random_y[n_id],
                        current.cost + d, c_id)

            if n_id in closedset:
                continue
            # Otherwise if it is already in the open set
            if n_id in openset:
                if openset[n_id].cost > node.cost:
                    openset[n_id].cost = node.cost
                    openset[n_id].pind = c_id
            else:
                openset[n_id] = node

    # generate final course
    rx, ry = [ngoal.x], [ngoal.y]
    pind = ngoal.pind
    while pind != -1:
        n = closedset[pind]
        rx.append(n.x)
        ry.append(n.y)
        pind = n.pind

    return rx, ry


def PRM_planning(x_start, y_start, x_goal, y_goal):

    road_map = generate_roadmap(random_x, random_y)

    rx, ry = dijkstra_planning(
        x_start, y_start, x_goal, y_goal, road_map, random_x, random_y)

    return rx, ry





random_x, random_y = random_points(x_start, y_start, x_goal, y_goal)
pl.scatter(random_x, random_y, s = 5)

rx, ry = PRM_planning(x_start, y_start, x_goal, y_goal)

assert rx, 'Not enough random point to find path'

pl.plot(rx, ry, "-r")
pl.show(block=True)
