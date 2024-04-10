import numpy as np
from scipy.interpolate import splprep, splev
from scipy.spatial.distance import cdist
from random import random
from copy import deepcopy
import pickle as pkl

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from shared import *




# nextPt = np.array([ random()*xSize, random()*ySize, -ii*dropPerSegment ])
PATH_ITERS = 100

# Generate initial path
path = randomPath(POINT_COUNT, BOUNDING_BOX)
targetHeights = -PT_DROP * np.arange(POINT_COUNT)

# Move points to local minima
for pathIteration in range(PATH_ITERS):

    # Init output forces
    forceSet = np.zeros_like(path)

    # Pull towards bounding box
    boundingBoxForce = pushTowardsBoundingBox(path, BOUNDING_BOX, 10, 1.0)

    # Pull towards Z position
    targHeightForce = pullTowardsTargetHeights(path, targetHeights, 0.1, 1)

    # Normalize distances between points50
    pathNormForce = normalizePathDists(path, PT_SPACING, 0.2)

    # Repel away from own path
    noSelfIntersectionForce = repelPathFromSelf(path, 2, 1.0, 10)

    # Limit path angle
    pathAngleForce = correctPathAngle(path, 2.7, 2.9, 1.0)[0]

    print("{:4.10f} {:4.10f} {:4.10f} {:4.10f} {:4.10f}".format(
        np.average(magnitude(boundingBoxForce)),
        np.average(magnitude(targHeightForce)),
        np.average(magnitude(pathNormForce)),
        np.average(magnitude(noSelfIntersectionForce)),
        np.average(magnitude(pathAngleForce)),
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

    path += boundingBoxForce + targHeightForce + pathNormForce + noSelfIntersectionForce + pathAngleForce




# Use spline interpolation for additonal points
bridgePoints = subdividePath(path)

print(f"X max{np.max(path[0])}")
print(f"X min{np.min(path[0])}")
print(f"Y max{np.max(path[1])}")
print(f"Y min{np.min(path[1])}")
print(f"Z max{np.max(path[2])}")
print(f"Z min{np.min(path[2])}")

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









ax = plt.figure().add_subplot(projection='3d')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

if True:
    ax.scatter(*path, color='blue')
    # ax.plot(*path, alpha=0.2, color='blue')

    forceSet = correctPathAngle(path, 2.5, 3, 5)[1]

    for idx in range(len(path[0])):
        pt = path[:, idx]
        vect = forceSet[:, idx]
        ax.plot(*np.swapaxes([pt, vect+pt], 0, 1), color='orange')

    ax.scatter(*bridgePoints[:, 1::2], color='purple')
    ax.plot(*bridgePoints, alpha=0.2, color='blue')

plt.show()

# print(path)
exit()
