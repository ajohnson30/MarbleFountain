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





# Check if the directory exists
if not os.path.exists(WORKING_DIR):
    # Create the directory
    os.makedirs(WORKING_DIR)


targHeightFracs = np.interp(
    np.linspace(0.0, 1.0, POINT_COUNT),
    [0.0, 0.9, 1.0],
    [1.0, 0.075, 0.0]
)
targetHeights = SIZE_Z * targHeightFracs

if LOAD_EXISTING_PATH:
    pathList = pkl.load(open(WORKING_DIR+'path.pkl', 'rb'))
    for idx in range(len(pathList)):
        pathList[idx] = pathList[idx][:, ::2]
else:
    # Generate initial path
    pathList = []
    for pathIdx in range(PATH_COUNT):
        path = randomPath(POINT_COUNT, BOUNDING_BOX)
        path[2, :] = targetHeights[:POINT_COUNT]
        pathList.append(path)

# Calculate set points at start and end of track
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
    setPoints[3, LOCKED_PT_CNT-2:LOCKED_PT_CNT+2] = 0.2
    setPoints[3, LOCKED_PT_CNT-1:LOCKED_PT_CNT+1] = 0.1

    # Init first and last points
    angle = getPathAnchorAngle(pathIdx)
    
    # pointRads = np.linspace(SCREW_RAD+PT_SPACING, (LOCKED_PT_CNT+1)*PT_SPACING + SCREW_RAD, LOCKED_PT_CNT)
    pointRads = np.arange(LOCKED_PT_CNT)*PT_SPACING + (SCREW_RAD + PT_SPACING)
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

if REALTIME_PLOTTING_FORCEMAGS:
    # Start the asynchronous plotting thread
    medianForceMagQueue = Queue()
    forceMagThread = threading.Thread(target=data_processor_and_plotter, args=(medianForceMagQueue,))
    forceMagThread.daemon = True
    forceMagThread.start()

if REALTIME_PLOTTING_PATHS:
    pathQueue = Queue()
    pathPlottingThread = threading.Thread(target=plot_paths_real_time, args=(pathQueue,))
    pathPlottingThread.daemon = True
    pathPlottingThread.start()



# Move points to local minima
for pathIteration in range(PATH_ITERS):
    pathFrac = pathIteration/PATH_ITERS
    moveMult = np.interp(
        [pathIteration/PATH_ITERS],
        [0.0, 0.5, 1.0],
        [1.0, 1.0, 0.1]
    )[0]

    randNoiseFactor = np.interp(
        [pathIteration/PATH_ITERS],
        [0.0, 0.1, 0.6, 0.9, 1.0],
        [20.0, 10.0, 2.0, 0.0, 0.0]
    )[0]

    if LOAD_EXISTING_PATH:
        randNoiseFactor = 0.0

    # randNoiseFactor = np.interp(
    #     [pathIteration/PATH_ITERS],
    #     [0.0, 1.0],
    #     [1.0, 0.0]
    # )[0]

    for pathIdx in range(len(pathList)):
        path = pathList[pathIdx]

        # Randomize points a little to add noise to help settle
        randNoiseArray = np.random.rand(*path.shape)
        randNoiseArray = randNoiseFactor * (2*randNoiseArray - 1.0)
        path += randNoiseArray

        # Init output forces
        forceSet = np.zeros_like(path)

        # Pull towards bounding box
        boundingBoxForce = pushTowardsBoundingBox(path, BOUNDING_BOX, 30.0, 2.0, axCount=3)
        if APPLY_FORCES_SEPARATELY: path += boundingBoxForce * moveMult

        # Pull towards Z position
        targHeightForce = pullTowardsTargetHeights(path, targetHeights[:path.shape[1]], 0.5, 5)
        # targHeightForce += pullTowardsTargetSlope(path, -PT_DROP, 0.3, 1.0)

        if APPLY_FORCES_SEPARATELY: path += targHeightForce * moveMult

        # Normalize distances between points
        # pathXY = deepcopy(path)
        # pathXY[2] = 0.0
        # pathNormForce = normalizePathDists(pathXY, PT_SPACING, 10.0)

        # pathNormForce = normalizePathDists(path,  PT_SPACING, 1.0, maxForce=10.0)

        pathNormForce = normalizePathDists(path,  PT_SPACING, 0.5, maxForce=10.0)
        pathNormForce += normalizePathDists(path,  PT_SPACING*2, 0.3, maxForce=10.0, pointOffset=2)
        pathNormForce += normalizePathDists(path,  PT_SPACING*2, 0.2, maxForce=10.0, pointOffset=3)
        
        # Apply part of each normalization force to adjacent points
        adjMix = 0.0
        for idx in range(pathNormForce.shape[1]):
            kernel_size = 5
            sigma = 1.0
            smoothed_data = weighted_average_convolution(pathNormForce[0], kernel_size, sigma)
            pathNormForce[0] = (1.0-adjMix)*pathNormForce[0] + adjMix*smoothed_data

        # plt.plot(smoothed_data, label='smoothed_data')
        # plt.plot(pathNormForce[0], label='pathNormForce[0]')
        # plt.legend()
        # plt.show()

        # pathNormForce = np.zeros_like(pathNormForce)
        if APPLY_FORCES_SEPARATELY: path += pathNormForce * moveMult

        # Repel away from own path
        noSelfIntersectionForce = repelPathFromSelf(path, 1, 30, ABSOLUTE_MIN_PT_DIST*6)
        noSelfIntersectionForce[:2] /= 4
        noSelfIntersectionForce = repelPathFromSelf(path, 3, 0.1, 40)
        if APPLY_FORCES_SEPARATELY: path += noSelfIntersectionForce * moveMult

        # Limit path angle
        pathAngleForce = correctPathAngle(path, 2.9, 3.14, 1.5)
        pathAngleForce += correctPathAngle(path, 3.1, 3.14, 0.1)
        pathAngleForce += correctPathAngle(path, 2.8, 3.14, 3.0, diffPointOffsetCnt=2)
        # pathAngleForceTest = correctPathAngle(path, 2.7, 3.14, 0.3, diffPointOffsetCnt=2) # Apply smoothing function across more points
        # pathAngleForce = correctPathAngle(path, 2.6, 3.14, 0.5, diffPointOffsetCnt=3) # Apply smoothing function across more points
        # pathAngleForce = correctPathAngle(path, 1.0, 3.1, 1.5, diffPointOffsetCnt=3) # Apply smoothing function across more points
        # pathAngleForce = correctPathAngle(path, 3.0, 3.1, 0.1, flatten=False)
        # pathAngleForce = correctPathAngle(path, 2.4, 3.0, 0.1, diffPointOffsetCnt=3) # Apply smoothing function across more points
        if APPLY_FORCES_SEPARATELY: path += pathAngleForce * moveMult

        # Repel away from other paths
        repelForce = np.zeros_like(path)
        for cmpIdx in range(len(pathList)):
            if pathIdx == cmpIdx: continue
            absoluteMinPathForce = repelPoints(path, pathList[cmpIdx], 5.0, ABSOLUTE_MIN_PT_DIST*1.5) # Absolute required distance between points, only inpacts Z
            absoluteMinPathForce[:2] /= 20.0
            repelForce += absoluteMinPathForce

            repelForce += repelPoints(path, pathList[cmpIdx], 2.0, ABSOLUTE_MIN_PT_DIST*3) # Absolute required distance between points
            # repelForce[2] = np.clip(repelForce[2], -5, 5)
            repelForce += repelPoints(path, pathList[cmpIdx], 0.001, 30)
        
        # Repel away from center lift
        repelForce += repelPoints(path, centerPoints, 4.0, 6*MARBLE_RAD+SCREW_RAD)
        if APPLY_FORCES_SEPARATELY: path += repelForce * moveMult

        # Do not slope up ever
        whereZincreases = np.where(path[2, 1:] > path[2, :-1])[0]
        path[2, whereZincreases] = path[2, whereZincreases+1]

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
            if not APPLY_FORCES_SEPARATELY:
                path += sumForce * APPLY_FORCES_SEPARATELY

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
            sumForce *= moveDistLOAD_EXISTING_PATH
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

            medianForceMags = [
                np.median(magnitude(boundingBoxForce)),
                np.median(magnitude(targHeightForce)),
                np.median(magnitude(pathNormForce)),
                np.median(magnitude(noSelfIntersectionForce)),
                np.median(magnitude(pathAngleForce)),
                np.median(magnitude(repelForce)),
            ]
            print("{:4d} {:1.3f} {:1.3f} {:4.10f} {:4.10f} {:4.10f} {:4.10f} {:4.10f} {:4.10f}".format(
                pathIteration,
                randNoiseFactor,
                moveMult,
                *medianForceMags
                ))

            if REALTIME_PLOTTING_FORCEMAGS:
                medianForceMagQueue.put({
                    'boundingBoxForce':medianForceMags[0],
                    'targHeightForce':medianForceMags[1],
                    'pathNormForce':medianForceMags[2],
                    'noSelfIntersectionForce':medianForceMags[3],
                    'pathAngleForce':medianForceMags[4],
                    'repelForce':medianForceMags[5],
                })

            if REALTIME_PLOTTING_PATHS and pathIteration%5 == 0:
                pathQueue.put(pathList)
            
            if SET_ITERATION_MOVE_DISTS:
                print("{:4.5f}: ".format(moveDist), end='')





    if pathIteration%5 == 0:
        # Use spline interpolation for additonal points
        fullPaths = [subdividePath(path) for path in pathList]
        # fullPaths = [path for path in pathList]

        pkl.dump(fullPaths, open(WORKING_DIR+'path.pkl', 'wb'))

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




# ax = plt.figure().add_subplot(projection='3d')
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# for pathIdx in range(len(pathList)):
#     bridgePoints = fullPaths[pathIdx]
#     path = pathList[pathIdx]


#     # ax.scatter(*centerPoints)

#     ax.scatter(*path)

#     # ax.scatter(*bridgePoints[:, 1::2], color='purple')
#     # ax.plot(*bridgePoints, alpha=0.5)

#     # ax.scatter(*path[:, :LOCKED_PT_CNT], color='red')
#     # ax.scatter(*path[:, -LOCKED_PT_CNT:], color='red')

#     if False:    
#         forceSet = correctPathAngle(path, 2.5, 3, 5)
#         for idx in range(len(path[0])):
#             pt = path[:, idx]
#             vect = forceSet[:, idx]
#             ax.plot(*np.swapaxes([pt, vect+pt], 0, 1), color='orange')


#     # visForce = normalizePathDists(path,  PT_SPACING, 1.0, maxForce=10.0)*5
#     visForce = correctPathAngle(path, 2.5, 3.14, 1.5, diffPointOffsetCnt=2)
#     for idx in range(path.shape[1]):
#         pt = path[:, idx]
#         vect = visForce[:, idx]
#         ax.plot(*np.swapaxes([pt, vect+pt], 0, 1), color='orange')


    fullPaths