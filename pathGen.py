import numpy as np
from scipy.interpolate import splprep, splev
from scipy.spatial.distance import cdist
from random import random
from copy import deepcopy
import pickle as pkl

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import os

from defs import *
from shared import *



ax = plt.figure().add_subplot(projection='3d')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')


# nextPt = np.array([ random()*xSize, random()*ySize, -ii*dropPerSegment ])

targetHeights = SIZE_Z - np.arange(0, SIZE_Z, SIZE_Z/POINT_COUNT)

# Generate initial path
pathList = []
for pathIdx in range(PATH_COUNT):
    path = randomPath(INIT_PATH_PTS, BOUNDING_BOX)
    path[2, :] = targetHeights[:INIT_PATH_PTS]

    # Init first and last points
    angle = np.pi*2*pathIdx/PATH_COUNT
    pointRads = np.linspace(0.0, LOCKED_PT_CNT*PT_SPACING, LOCKED_PT_CNT)
    path[0, :LOCKED_PT_CNT] = np.cos(angle)*pointRads + SIZE_X/2
    path[1, :LOCKED_PT_CNT] = np.sin(angle)*pointRads + SIZE_Y/2
    path[0, -LOCKED_PT_CNT:] = np.flip(np.cos(angle)*pointRads) + SIZE_X/2
    path[1, -LOCKED_PT_CNT:] = np.flip(np.sin(angle)*pointRads) + SIZE_Y/2

    pathList.append(path)

# Calculate no go points
centerPoints = np.arange(0, SIZE_Z, PT_SPACING)
centerPoints = np.array([np.zeros_like(centerPoints), np.zeros_like(centerPoints), centerPoints])
centerPoints[0, :] = SIZE_X/2
centerPoints[1, :] = SIZE_Y/2

# Move points to local minima
for pathIteration in range(PATH_ITERS):
    for pathIdx in range(len(pathList)):
        path = pathList[pathIdx]
        
        # Init output forces
        forceSet = np.zeros_like(path)

        # Pull towards bounding box
        boundingBoxForce = pushTowardsBoundingBox(path, BOUNDING_BOX, 100.0, 5.0, axCount=3)

        # Pull towards Z position
        targHeightForce = pullTowardsTargetHeights(path, targetHeights[:path.shape[1]], 0.05, 5)

        # Normalize distances between points
        pathNormForce = normalizePathDists(path, PT_SPACING, 0.2)

        # Repel away from own path
        noSelfIntersectionForce = repelPathFromSelf(path, 2, 10, ABSOLUTE_MIN_PT_DIST)
        noSelfIntersectionForce = repelPathFromSelf(path, 10, 0.01, 100)

        # Limit path angle
        pathAngleForce = correctPathAngle(path, 3.0, 3.2, 1.0)

        # Repel away from other paths
        repelForce = np.zeros_like(path)
        for cmpIdx in range(len(pathList)):
            if pathIdx == cmpIdx: continue
            repelForce += repelPoints(path, pathList[cmpIdx], 20, ABSOLUTE_MIN_PT_DIST*2) # Absolute required distance between points
            # repelForce[2] = np.clip(repelForce[2], -5, 5)
            repelForce += repelPoints(path, pathList[cmpIdx], 0.01, 50)
        
        # Repel away from center lift
        repelForce += repelPoints(path, centerPoints, 5.0, ABSOLUTE_MIN_PT_DIST+SCREW_RAD)

        
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

        sumForce = boundingBoxForce + targHeightForce + pathNormForce + noSelfIntersectionForce + pathAngleForce + repelForce

        if pathIteration in RESAMPLE_AT:
            forceMag = magnitude(sumForce)
            path[:, LOCKED_PT_CNT:-LOCKED_PT_CNT] = redistributePathByForce(path[:, LOCKED_PT_CNT:-LOCKED_PT_CNT], sumForce[:, LOCKED_PT_CNT:-LOCKED_PT_CNT])
            # path = redistributePathByForce(path, sumForce)
        else:
            path[:, LOCKED_PT_CNT:-LOCKED_PT_CNT] += sumForce[:, LOCKED_PT_CNT:-LOCKED_PT_CNT]


        # Constant dist moves that gradually converge
        if SET_ITERATION_MOVE_DISTS:
            sumForce /= magnitude(sumForce)
            moveDist = 10.0 * np.square((PATH_ITERS - pathIteration)/PATH_ITERS)
            sumForce *= moveDist

                
        if False:
            # Add points gradually
            if path.shape[1] < POINT_COUNT:
                newPt = path[:, -1:]

                newPt[0, 0] += PT_DROP - np.random.random()*2*PT_DROP
                newPt[1, 0] += PT_DROP - np.random.random()*2*PT_DROP
                newPt[2, 0] -= PT_DROP
                
                path = np.concatenate([path, newPt], axis=1)

                # print(path.shape)
        
        # Handle any consequtive points overlapping in XY, as this causes a divide by 0
        singularities = np.where(magnitude(path[:2, 1:] - path[:2, :-1]) < 0.1)
        path[0, singularities] += 0.01

        pathList[pathIdx] = path

        if pathIdx == 0:

            print("{:4.10f} {:4.10f} {:4.10f} {:4.10f} {:4.10f} {:4.10f}".format(
                np.median(magnitude(boundingBoxForce)),
                np.median(magnitude(targHeightForce)),
                np.median(magnitude(pathNormForce)),
                np.median(magnitude(noSelfIntersectionForce)),
                np.median(magnitude(pathAngleForce)),
                np.median(magnitude(repelForce)),
                ))
            
            if SET_ITERATION_MOVE_DISTS:
                print("{:4.5}: ".format(moveDist), end='')

        
# Use spline interpolation for additonal points
fullPaths = [subdividePath(path) for path in pathList]
# fullPaths = [path for path in pathList]

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






for pathIdx in range(len(pathList)):
    bridgePoints = fullPaths[pathIdx]
    path = pathList[pathIdx]


    # ax.scatter(*centerPoints)

    # ax.scatter(*path)

    # ax.scatter(*path[:, :LOCKED_PT_CNT], color='red')
    # ax.scatter(*path[:, -LOCKED_PT_CNT:], color='red')

    if False:    
        forceSet = correctPathAngle(path, 2.5, 3, 5)
        for idx in range(len(path[0])):
            pt = path[:, idx]
            vect = forceSet[:, idx]
            ax.plot(*np.swapaxes([pt, vect+pt], 0, 1), color='orange')


    # for idx in range(path.shape[1]):
    #         pt = path[:, idx]
    #         vect = forceSet[:, idx] * 10
    #         ax.plot(*np.swapaxes([pt, vect+pt], 0, 1), color='orange')


    # ax.scatter(*bridgePoints[:, 1::2], color='purple')
    ax.plot(*bridgePoints, alpha=0.5)
    
    ax.set_aspect('equal', adjustable='box')


# Check if the directory exists
if not os.path.exists(WORKING_DIR):
    # Create the directory
    os.makedirs(WORKING_DIR)

pkl.dump(fullPaths, open(WORKING_DIR+'path.pkl', 'wb'))

plt.show()