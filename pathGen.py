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

targetHeights = SIZE_Z - np.linspace(0, SIZE_Z, POINT_COUNT)

# Generate initial path
pathList = []
for pathIdx in range(PATH_COUNT):
    path = randomPath(POINT_COUNT, BOUNDING_BOX)
    path[2, :] = targetHeights[:POINT_COUNT]
    pathList.append(path)

# Generate set points at start and end of track
setPointIndexList = []
setPointList = []
for pathIdx in range(PATH_COUNT):
    # Calculate 
    setPointIndices = np.zeros(LOCKED_PT_CNT*2, dtype=np.int32)
    setPointIndices[:LOCKED_PT_CNT] = np.arange(LOCKED_PT_CNT)
    setPointIndices[-LOCKED_PT_CNT:] = np.arange(POINT_COUNT-LOCKED_PT_CNT, POINT_COUNT)

    setPoints = np.zeros((4, LOCKED_PT_CNT*2), dtype=np.double)

    # forceDecay = 2 * (np.cos(np.linspace(0.0, np.pi/3, LOCKED_PT_CNT+1))[:-1] - 0.5)
    # setPoints[3, :LOCKED_PT_CNT] = forceDecay
    # setPoints[3, -LOCKED_PT_CNT:] = np.flip(forceDecay)
    setPoints[3] = 1.0
    setPoints[3, LOCKED_PT_CNT:LOCKED_PT_CNT+2] = 0.1

    # Init first and last points
    angle = np.pi*2*pathIdx/PATH_COUNT
    pointRads = np.linspace(SCREW_RAD+PT_SPACING, LOCKED_PT_CNT*PT_SPACING + SCREW_RAD, LOCKED_PT_CNT)
    setPoints[0, :LOCKED_PT_CNT] = np.cos(angle)*pointRads + SIZE_X/2
    setPoints[1, :LOCKED_PT_CNT] = np.sin(angle)*pointRads + SIZE_Y/2
    setPoints[0, -LOCKED_PT_CNT:] = np.flip(np.cos(angle)*pointRads) + SIZE_X/2
    setPoints[1, -LOCKED_PT_CNT:] = np.flip(np.sin(angle)*pointRads) + SIZE_Y/2
    setPoints[2] = targetHeights[setPointIndices]

    # Save points
    setPointIndexList.append(setPointIndices)
    setPointList.append(setPoints)

# Calculate no go points
centerPoints = np.arange(0, SIZE_Z, PT_SPACING)
centerPoints = np.array([np.zeros_like(centerPoints), np.zeros_like(centerPoints), centerPoints])
centerPoints[0, :] = SIZE_X/2
centerPoints[1, :] = SIZE_Y/2

# Move points to local minima
for pathIteration in range(PATH_ITERS):
    pathFrac = pathIteration/PATH_ITERS

    for pathIdx in range(len(pathList)):
        path = pathList[pathIdx]
        
        # Init output forces
        forceSet = np.zeros_like(path)

        # Pull towards bounding box
        boundingBoxForce = pushTowardsBoundingBox(path, BOUNDING_BOX, 30.0, 2.0, axCount=3)

        # Pull towards Z position
        targHeightForce = pullTowardsTargetHeights(path, targetHeights[:path.shape[1]], 0.5, 5)
        # targHeightForce += pullTowardsTargetSlope(path, -PT_DROP, 0.3, 1.0)

        # Normalize distances between points
        # pathXY = deepcopy(path)
        # pathXY[2] = 0.0
        # pathNormForce = normalizePathDists(pathXY, PT_SPACING, 10.0)

        pathNormForce = normalizePathDists(path,  PT_SPACING, 1.0)
        # pathNormForce = np.zeros_like(pathNormForce)

        # Repel away from own path
        noSelfIntersectionForce = repelPathFromSelf(path, 3, 10, ABSOLUTE_MIN_PT_DIST*2)
        noSelfIntersectionForce = repelPathFromSelf(path, 5, 0.01, 10)

        # Limit path angle
        pathAngleForce = correctPathAngle(path, 2.9, 3.14, 1.0)
        # pathAngleForce = correctPathAngle(path, 2.0, 3.1, 1.5, diffPointOffsetCnt=2) # Apply smoothing function across more points
        # pathAngleForce = correctPathAngle(path, 1.0, 3.1, 0.5, diffPointOffsetCnt=3) # Apply smoothing function across more points
        # pathAngleForce = correctPathAngle(path, 3.0, 3.1, 0.1, flatten=False)
        # pathAngleForce = correctPathAngle(path, 2.4, 3.0, 0.1, diffPointOffsetCnt=3) # Apply smoothing function across more points

        # Repel away from other paths
        repelForce = np.zeros_like(path)
        for cmpIdx in range(len(pathList)):
            if pathIdx == cmpIdx: continue
            repelForce += repelPoints(path, pathList[cmpIdx], 10.0, ABSOLUTE_MIN_PT_DIST) # Absolute required distance between points
            repelForce += repelPoints(path, pathList[cmpIdx], 2.5, ABSOLUTE_MIN_PT_DIST*3) # Absolute required distance between points
            # repelForce[2] = np.clip(repelForce[2], -5, 5)
            # repelForce += repelPoints(path, pathList[cmpIdx], 0.01, 20)
        
        # Repel away from center lift
        repelForce += repelPoints(path, centerPoints, 4.0, 2*ABSOLUTE_MIN_PT_DIST+SCREW_RAD)

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
            print(f"   ----- RESAMPLING -----")
            forceMag = magnitude(sumForce)
            path = redistributePathByForce(path, sumForce)
            # path = redistributePathByForce(path, sumForce)
        else:
            path += sumForce

            # Pull towards set points
            setPointIndices = setPointIndexList[pathIdx]
            setPoints = setPointList[pathIdx]
            setPtForces = setPoints[3] * (setPoints[:3] - path[:, setPointIndices])
            
            path[:, setPointIndices] += setPtForces

            if False: # Do no go up
                diff = np.diff(path[2])
                inc = np.where(diff > 0)
                path[2, 1:][inc] = path[2, :-1][inc]

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
                print("{:4.5f}: ".format(moveDist), end='')

        
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

    ax.scatter(*path)

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