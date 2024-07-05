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


startPoints = 3
targetHeights = np.zeros(POINT_COUNT, dtype=np.double)
# Set initial points
INITIAL_POINT_MULT = 2
targetHeights[:startPoints] = SIZE_Z - INITIAL_POINT_MULT*PT_DROP*np.arange(startPoints)
targetHeights[-startPoints:] = INITIAL_POINT_MULT*PT_DROP*(startPoints-1-np.arange(startPoints))

# Interpolate rest of points
targetHeights[startPoints:-startPoints] = (SIZE_Z - INITIAL_POINT_MULT*PT_DROP*2*(startPoints+1))*np.interp(
    np.linspace(0.0, 1.0, POINT_COUNT - 2*startPoints),
    [0.0, 0.9, 1.0],
    [1.0, 0.06, 0.0]
) + INITIAL_POINT_MULT*PT_DROP*(startPoints+1)


# Init path points
if LOAD_EXISTING_PATH and os.path.exists(WORKING_DIR+'path.pkl'):
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

if False:
    diffPointOffsetCnt = 1

    path = pathList[0]
    centerVect, radii = approximatePathCurvature(path, diffPointOffsetCnt)
    plt.plot(1/radii)

    # Calculate vectors and normals to preceding and succeeding point
    pathDiffs = path[:, diffPointOffsetCnt:] - path[:, :-diffPointOffsetCnt]
    nextPtVect = pathDiffs[:, diffPointOffsetCnt:]
    prevPtVect = -pathDiffs[:, :-diffPointOffsetCnt]

    prevNorm = prevPtVect/magnitude(prevPtVect)
    nextNorm = nextPtVect/magnitude(nextPtVect)

    # Calculate angle between prev and next vector
    dotProducts = np.zeros(nextPtVect.shape[1])
    for idx in range(len(dotProducts)): 
        dotProducts[idx] = np.dot(nextNorm[:, idx], prevNorm[:, idx])

    angles = np.arccos(np.clip(dotProducts, -1.0, 1.0))

    plt.plot(angles/np.pi)




    plt.show()
    exit()


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

# Init path temperatures
pathTempList = np.array([0.0 for idx in range(len(pathList))])

# Iteratively calculate path
for pathIteration in range(PATH_ITERS):
    pathFrac = pathIteration/PATH_ITERS

    # pathTempList[:] = np.max(pathTempList)

    if DO_DYNAMIC_TEMPERATURE:
        print(f"{pathIteration:>4}/{PATH_ITERS} : ", end='')
    else:
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
        
        if DO_DYNAMIC_TEMPERATURE:
            randNoiseFactor = np.clip(pathTempList[pathIdx], 0.0, np.inf)
            moveMult = (10.0 + np.clip(pathTempList[pathIdx], -10.0, 0.0)) / 10.0
            # moveMult *= 0.5

        # Randomize points a little to add noise to help settle
        randNoiseArray = np.random.rand(*path.shape)
        randNoiseArray = randNoiseFactor * (2*randNoiseArray - 1.0)
        path += randNoiseArray

        # Init output force list
        # boundingBoxForce, targHeightForce, pathNormForce, noSelfIntersectionForce, pathAngleForce, repelForce, downHillForce, changeInSlopeForce, setPtForce
        forceList = []

        # Pull towards bounding box
        boundingBoxForce = pushTowardsBoundingBox(path, BOUNDING_BOX, 30.0, 2.0, axCount=3)
        if APPLY_FORCES_SEPARATELY: path += boundingBoxForce * moveMult
        forceList.append(boundingBoxForce)

        # Pull towards Z position
        if not GLASS_MARBLE_14mm:
            targHeightForce = pullTowardsTargetHeights(path, targetHeights[:path.shape[1]], 0.15, 5)
        else:
            targHeightForce = pullTowardsTargetHeights(path, targetHeights[:path.shape[1]], 0.5, 5)
        # targHeightForce += pullTowardsTargetSlope(path, -PT_DROP, 0.3, 1.0)
        if APPLY_FORCES_SEPARATELY: path += targHeightForce * moveMult
        forceList.append(targHeightForce)

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
        kernel_size = 5
        sigma = 1.0
        for idx in range(pathNormForce.shape[1]):
            smoothed_data = weighted_average_convolution(pathNormForce[0], kernel_size, sigma)
            pathNormForce[0] = (1.0-adjMix)*pathNormForce[0] + adjMix*smoothed_data


        # plt.plot(smoothed_data, label='smoothed_data')
        # plt.plot(pathNormForce[0], label='pathNormForce[0]')
        # plt.legend()
        # plt.show()

        # pathNormForce = np.zeros_like(pathNormForce)
        if APPLY_FORCES_SEPARATELY: path += pathNormForce * moveMult
        forceList.append(pathNormForce)

        # Repel away from own path
        noSelfIntersectionForce = repelPathFromSelf(path, 2, 2.0, ABSOLUTE_MIN_PT_DIST*2.0)
        noSelfIntersectionForce[:2] /= 8
        noSelfIntersectionForce += repelPathFromSelf(path, 4, 0.02, 40)
        if APPLY_FORCES_SEPARATELY: path += noSelfIntersectionForce * moveMult
        forceList.append(noSelfIntersectionForce)

        # Limit path angle
        if False:
            if not GLASS_MARBLE_14mm:
                pathAngleForce = correctPathAngle(path, 2.9, 3.14, 1.5)
                # pathAngleForce += correctPathAngle(path, 3.1, 3.14, 0.1)
                pathAngleForce += correctPathAngle(path, 2.6, 3.1, 3.0, diffPointOffsetCnt=2)
                pathAngleForce += correctPathAngle(path, 2.0, 3.0, 2.0, diffPointOffsetCnt=3)
                pathAngleForce += correctPathAngle(path, 0.5, 3.0, 1.0, diffPointOffsetCnt=4)
            else:
                pathAngleForce = correctPathAngle(path, 2.9, 3.14, 1.5)
                # pathAngleForce += correctPathAngle(path, 3.1, 3.14, 0.1)
                pathAngleForce += correctPathAngle(path, 2.6, 3.14, 3.0, diffPointOffsetCnt=2)

        # Limit path curvature
        if True:
            pathAngleForce = tempCurvatureCalc(path)

        # pathAngleForceTest = correctPathAngle(path, 2.7, 3.14, 0.3, diffPointOffsetCnt=2) # Apply smoothing function across more points
        # pathAngleForce = correctPathAngle(path, 2.6, 3.14, 0.5, diffPointOffsetCnt=3) # Apply smoothing function across more points
        # pathAngleForce = correctPathAngle(path, 1.0, 3.1, 1.5, diffPointOffsetCnt=3) # Apply smoothing function across more points
        # pathAngleForce = correctPathAngle(path, 3.0, 3.1, 0.1, flatten=False)
        # pathAngleForce = correctPathAngle(path, 2.4, 3.0, 0.1, diffPointOffsetCnt=3) # Apply smoothing function across more points
        if APPLY_FORCES_SEPARATELY: path += pathAngleForce * moveMult
        forceList.append(pathAngleForce)

        # Repel away from other paths
        repelForce = np.zeros_like(path)
        for cmpIdx in range(len(pathList)):
            if pathIdx == cmpIdx: continue
            absoluteMinPathForce = repelPoints(path, pathList[cmpIdx], 5.0, ABSOLUTE_MIN_PT_DIST*1.5) # Absolute required distance between points, only inpacts Z
            absoluteMinPathForce[:2] /= 20.0
            repelForce += absoluteMinPathForce

            repelForce += repelPoints(path, pathList[cmpIdx], 1.0, ABSOLUTE_MIN_PT_DIST*3) # Required distance between points

            repelForce += repelPoints(path, pathList[cmpIdx], 0.001, 30) # Broadly avoid other paths
            repelForce += repelPoints(path, pathList[cmpIdx][:, [0, -1]], 2.0, 30) # Avoid end points of other paths
            
        
        # Repel away from center lift
        repelForce += repelPoints(path, centerPoints, 4.0, 6*MARBLE_RAD+SCREW_RAD)
        if APPLY_FORCES_SEPARATELY: path += repelForce * moveMult
        forceList.append(repelForce)

        # Do not slope up ever
        downHillForce = preventUphillMotion(path, 1.0)
        if APPLY_FORCES_SEPARATELY: path += downHillForce * moveMult
        forceList.append(downHillForce)


        # zPoints = deepcopy(path[2])
        # for idx in range(0, len(zPoints) - 1):
        #     if zPoints[idx] < zPoints[idx + 1]:
        #         zPoints[idx] = zPoints[idx + 1]
        # slopeDownForce[2] = zPoints - path[2]
        
        
        # zDiff = np.diff(path[2])
        # zDiff[zDiff < 0.0] = 0.0
        # slopeDownForce[2, 1:] -= zDiff/2
        # slopeDownForce[2, :-1] += zDiff/2

        # whereZincreases = np.where(path[2, 1:] > path[2, :-1])[0]
        # path[2, whereZincreases] = path[2, whereZincreases+1]
        # if APPLY_FORCES_SEPARATELY: path += slopeDownForce # Not sloping up ignores moveMult
        # forceList.append(slopeDownForce)


        if not GLASS_MARBLE_14mm:
            changeInSlopeForce = correctSlopeChange(path, 0.5, 2.0)
        else:
            changeInSlopeForce = correctSlopeChange(path, 0.2, 0.5)
        if APPLY_FORCES_SEPARATELY: path += changeInSlopeForce * moveMult # Not sloping up ignores moveMult
        changeInSlopeForce = np.zeros_like(changeInSlopeForce)
        forceList.append(changeInSlopeForce)


        # Pull towards set points
        setPtForce = np.zeros_like(path)
        setPointIndices = setPointIndexList[pathIdx]
        setPoints = setPointList[pathIdx]
        setPtForce[:, setPointIndices] = setPoints[3] * (setPoints[:3] - path[:, setPointIndices])
        if APPLY_FORCES_SEPARATELY: path += setPtForce * moveMult
        forceList.append(setPtForce)

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

        # Constant dist moves that gradually converge
        if SET_ITERATION_MOVE_DISTS:
            sumForce = np.sum(forceList, axis=0)

            sumForce /= magnitude(sumForce)
            moveDist = 10.0 * np.square((PATH_ITERS - pathIteration)/PATH_ITERS)
            sumForce *= moveDist
            # Add points gradually
            if path.shape[1] < POINT_COUNT:
                newPt = path[:, -1:]

                newPt[0, 0] += PT_DROP - np.random.random()*2*PT_DROP
                newPt[1, 0] += PT_DROP - np.random.random()*2*PT_DROP
                newPt[2, 0] -= PT_DROP
                
                path = np.concatenate([path, newPt], axis=1)
        
        # Just sum forces
        elif not APPLY_FORCES_SEPARATELY:
            sumForce = np.sum(forceList, axis=0)
            path += sumForce * APPLY_FORCES_SEPARATELY


        # Calculate force data
        forceMags = [magnitude(fooFrc) for fooFrc in forceList]

        # Update dynamic temperature
        if DO_DYNAMIC_TEMPERATURE:
            forceSums = np.sum(forceMags[:-1], axis=0) # Sum forces excluding setpointforce
            maxForceIdx = np.argmax(forceSums)
            if False and pathIdx == 0:
                print("{:3.5f}".format(forceSums[maxForceIdx]), end=' | ')
                for fooForceMag in forceMags:
                    print("{:3.5f}".format(fooForceMag[maxForceIdx]), end=' ')
                print(' ')
            
            # Calculate current temperature at point
            maxSumForce = forceSums[maxForceIdx]
            temperatureVal = np.interp(maxSumForce, PATH_RANDOMIZATION_FUNC[0], PATH_RANDOMIZATION_FUNC[1])
            temperatureDrop = np.interp(pathTempList[pathIdx], PATH_RANDOMIZATION_FUNC[1], PATH_RANDOMIZATION_FUNC[2])
            pathTempList[pathIdx] -= temperatureDrop
            
            # Update if appropriate
            if temperatureVal > pathTempList[pathIdx]:
                pathTempList[pathIdx] = temperatureVal

            # Print data
            print(f"{maxSumForce:>8.5f} {pathTempList[pathIdx]:>8.5f}", end=' | ')
                
        # Resample path if requested
        if pathIteration in RESAMPLE_AT:
            sumForce = np.sum(forceList, axis=0)
            print(f"   ----- RESAMPLING -----")
            forceMag = magnitude(sumForce)
            path = redistributePathByForce(path, sumForce)
            # path = redistributePathByForce(path, sumForce)

        # Handle any consequtive points overlapping in XY, as this causes a divide by 0
        singularities = np.where(magnitude(path[:2, 1:] - path[:2, :-1]) < 0.1)
        path[0, singularities] += 0.01

        # Save map
        pathList[pathIdx] = path

        # Plot forces for first map
        if pathIdx == 0:
            if not DO_DYNAMIC_TEMPERATURE:
                medianForceMags = [np.median(fooMag) for fooMag in forceMags]
                print("{:4d} {:1.3f} {:1.3f} |".format(
                    pathIteration,
                    randNoiseFactor,
                    moveMult,
                    *medianForceMags
                    ), end=' ')
                
                for fooMedFrcMag in medianForceMags:
                    print("{:3.5f}".format(fooMedFrcMag), end=' ')

                if SET_ITERATION_MOVE_DISTS:
                    print("{:4.5f}: ".format(moveDist), end='')
                
                print(' ')

            if REALTIME_PLOTTING_FORCEMAGS:
                medianForceMags = [np.max(fooMag) for fooMag in forceMags]
                medianForceMagQueue.put({
                     'boundingBoxForce': medianForceMags[0],
                    'targHeightForce': medianForceMags[1],
                    'pathNormForce': medianForceMags[2],
                    'noSelfIntersectionForce': medianForceMags[3], 
                    'pathAngleForce': medianForceMags[4],
                    'repelForce': medianForceMags[5],
                    'downHillForce': medianForceMags[6],
                    'changeInSlopeForce': medianForceMags[7],
                    'setPtForce':medianForceMags[8],
                })

            if REALTIME_PLOTTING_PATHS and pathIteration%5 == 0:
                pathQueue.put(pathList)
            

    # Print endline for logging
    if DO_DYNAMIC_TEMPERATURE:
        print(' ')



    if pathIteration%5 == 0:
        # Use spline interpolation for additonal points
        fullPaths = [subdividePath(path) for path in pathList]
        # fullPaths = [path for path in pathList]

        pkl.dump(fullPaths, open(WORKING_DIR+'path.pkl', 'wb'))




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

