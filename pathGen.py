import numpy as np
from scipy.interpolate import splprep, splev
from scipy.spatial.distance import cdist
from random import random
from copy import deepcopy
import pickle as pkl

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys

from shared import *




# nextPt = np.array([ random()*xSize, random()*ySize, -ii*dropPerSegment ])
PATH_ITERS = 100
PATH_COUNT = 1


# Generate initial path
pathList = []
for pathIdx in range(PATH_COUNT):
    path = randomPath(POINT_COUNT, BOUNDING_BOX)
    pathList.append(path)

targetHeights = -PT_DROP * np.arange(POINT_COUNT)

# Move points to local minima
for pathIteration in range(PATH_ITERS):
    for pathIdx in range(len(pathList)):
        path = pathList[pathIdx]
        
        # Init output forces
        forceSet = np.zeros_like(path)

        # Pull towards bounding box
        boundingBoxForce = pushTowardsBoundingBox(path, BOUNDING_BOX, 10, 1.0)

        # Pull towards Z position
        targHeightForce = pullTowardsTargetHeights(path, targetHeights, 0.1, 1)

        # Normalize distances between points50
        pathNormForce = normalizePathDists(path, PT_SPACING, 0.2)

        # Repel away from own path
        noSelfIntersectionForce = repelPathFromSelf(path, 2, 1.0, 15)

        # Limit path angle
        pathAngleForce = correctPathAngle(path, 2.7, 3.1, 1.0)[0]

        # Repel away from other paths
        repelForce = np.zeros_like(path)
        for cmpIdx in range(len(pathList)):
            if pathIdx == cmpIdx: continue
            repelForce += repelPoints(path, pathList[cmpIdx], 0.5, 15)



        print("{:4.10f} {:4.10f} {:4.10f} {:4.10f} {:4.10f}".format(
            np.average(magnitude(boundingBoxForce)),
            np.average(magnitude(targHeightForce)),
            np.average(magnitude(pathNormForce)),
            np.average(magnitude(noSelfIntersectionForce)),
            np.average(magnitude(pathAngleForce)),
            np.average(magnitude(repelForce)),
            ))


        if False:
            ax = plt.figure().add_subplot(projection='3d')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')

            ax.plot(*path, color='blue')
            ax.scatter(*path, color='blue')

            if True:
                ax.scatter(*path, color='blue')
                ax.plot(*path, alpha=0.2, color='blue')

                for idx in range(len(path[0])):
                    pt = path[:, idx]
                    vect = forceSet[:, idx]
                    ax.plot(*np.swapaxes([pt, vect+pt], 0, 1), color='orange')

            plt.show()

        path += boundingBoxForce + targHeightForce + pathNormForce + noSelfIntersectionForce + pathAngleForce + repelForce




# Use spline interpolation for additonal points
fullPaths = [subdividePath(path) for path in pathList]

# Generate supports
    # Normalize distances

    # Pull towards bounding box

    # Repel away from paths
    
    # Reduce overhangs

    # Add constant downwards pull

    # Attract dangling edges to other supports
    
    # Merge overlapping edges

    # End chains that hit ground

    # Add a new set of points to all dangling chains every few iterations


# FILE_NAME = 'default'
# if len(sys.argv) > 1:
# 	FILE_NAME = sys.argv[1]

# outFile = open(f'output/{FILE_NAME}/path.pkl', 'rb')






ax = plt.figure().add_subplot(projection='3d')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

for pathIdx in range(len(pathList)):
    bridgePoints = fullPaths[pathIdx]
    path = pathList[pathIdx]


    ax.scatter(*path)
    forceSet = correctPathAngle(path, 2.5, 3, 5)[1]

    # for idx in range(len(path[0])):
    #     pt = path[:, idx]
    #     vect = forceSet[:, idx]
    #     ax.plot(*np.swapaxes([pt, vect+pt], 0, 1), color='orange')


    for idx in range(path.shape[1]):
            pt = path[:, idx]
            vect = forceSet[:, idx] * 10
            ax.plot(*np.swapaxes([pt, vect+pt], 0, 1), color='orange')


    # ax.scatter(*bridgePoints[:, 1::2], color='purple')
    ax.plot(*bridgePoints, alpha=0.5)


plt.show()