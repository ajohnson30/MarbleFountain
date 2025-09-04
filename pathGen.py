import numpy as np
from scipy.interpolate import splprep, splev
from scipy.spatial.distance import cdist
from random import random
from copy import deepcopy
import pickle as pkl
import glob

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import os

from defs import *
from shared import *
import positionFuncs as pf

# Check if the directory exists
if not os.path.exists(WORKING_DIR):
    # Create the directory
    os.makedirs(WORKING_DIR)

# Check if the directory exists
if not os.path.exists(WORKING_DIR+"PathDump/"):
    os.makedirs(WORKING_DIR+"PathDump/")
    pathDumpIter = 1
else:
    # Get a list of all existing .pkl files in the folder
    existing_files = glob.glob(os.path.join(WORKING_DIR+"PathDump/", "*.pkl"))
    
    # Calculate the next iteration number
    if not existing_files:
        pathDumpIter = 0
    else:
        # Extract the highest iteration number from existing files
        highest_iteration = max([int(os.path.basename(f).split('.')[0]) for f in existing_files])
        pathDumpIter = highest_iteration + 1

print(f"SIZE_X:{SIZE_X}")
print(f"SIZE_Y:{SIZE_Y}")
print(f"SIZE_Z:{SIZE_Z}")


startPoints = 3
targetHeights = np.zeros(POINT_COUNT, dtype=np.double)
# Set initial points
# Set beginning and end of path to steep decline
inputOutputSlope = INITIAL_POINT_MULT_SLOPE*(np.arange(startPoints) + 1)
startAndEndOffset = INITIAL_POINT_MULT_SLOPE*(startPoints + 1)
targetHeights[:startPoints] = SIZE_Z - inputOutputSlope
targetHeights[-startPoints:] = np.flip(inputOutputSlope)

# Interpolate rest of points
targetHeights[startPoints:-startPoints] = (SIZE_Z - 2*startAndEndOffset)*np.interp(
    np.linspace(0.0, 1.0, POINT_COUNT - 2*startPoints),
    [0.0, 0.9, 1.0],
    [1.0, 0.1, 0.0]
) + startAndEndOffset


# Init path points
if LOAD_EXISTING_PATH and os.path.exists(WORKING_DIR+'path.pkl') and '-reset' not in sys.argv:
    pathList = pkl.load(open(WORKING_DIR+'path.pkl', 'rb'))
    initialTemperature = -5.0
    # initialTemperature = 5.0
else:
    # Generate initial path
    pathList = []
    for pathIdx in range(PATH_COUNT):
        path = randomPath(POINT_COUNT, BOUNDING_BOX, pathIdx)
        path[2, :] = targetHeights
        pathList.append(path)
    initialTemperature = 15.0

    pkl.dump(pathList, open(WORKING_DIR+'path.pkl', 'wb'))
    pkl.dump(pathList, open(WORKING_DIR+'PathDump/000000.pkl', 'wb'))

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
    pointRads = (np.arange(LOCKED_PT_CNT) + 1)*PT_SPACING + (SCREW_RAD + PT_SPACING)
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

# Init path temperatures
pathTempList = np.array([initialTemperature for idx in range(len(pathList))])
pathTempHistList = np.array([1.0e9 * np.ones(TEMPERATURE_HISTORY_LEN, dtype=np.double) for idx in range(len(pathList))])

# Iteratively calculate path
for pathIteration in range(PATH_ITERS):
    pathFrac = pathIteration/PATH_ITERS


    if np.sum(pathTempList) <= -10.0*PATH_COUNT or (MIRROR_PATHS and pathTempList[0] <= -10.0):
        print(f"Paths converged, exiting early")
        break

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

    for pathIdx in range(len(pathList)):
        path = pathList[pathIdx]
        
        if DO_DYNAMIC_TEMPERATURE:
            randNoiseFactor = np.clip(pathTempList[pathIdx], 0.0, np.inf)
            moveMult = (10.0 + np.clip(pathTempList[pathIdx], -10.0, 0.0)) / 10.0

            scaleFactA = np.interp(pathTempList[pathIdx], [0.0, 2.0, 5.0], [0.0, 0.1, 1.0])
            scaleFactB = np.interp(pathTempList[pathIdx], [2.0, 10.0, 15.0], [1.0, 0.1, 0.05])
            scaleFactC = np.interp(pathTempList[pathIdx], [-0.5, 0.0, 2.0], [1.0, 0.6, 0.0])
            scaleFactD = np.interp(pathTempList[pathIdx], [-9.0, -7.5], [1.0, 0.0])
        else:
            scaleFactB = 1.0
            scaleFactC = 1.0
            # moveMult *= 0.1

        # Randomize points a little to add noise to help settle
        randNoiseArray = np.random.rand(path.shape[0]-1, path.shape[1])
        randNoiseArray = randNoiseFactor * (2*randNoiseArray - 1.0)
        path[:2] += randNoiseArray

        # Init output force list
        # boundingBoxForce, targHeightForce, pathNormForce, noSelfIntersectionForce, pathAngleForce, repelForce, changeInSlopeForce, downHillForce, setPtForce
        forceList = []



        # Pull towards bounding box
        boundingBoxForceCurve = [[-10, 0, 5, 10], [0.0, 0.1, 5, 30.0]]
        boundingBoxForce = pushTowardsBoundingBox(path, BOUNDING_BOX, boundingBoxForceCurve, axCount=3)
        boundingBoxForce[np.abs(boundingBoxForce) > 0.1] *= scaleFactB
        path += boundingBoxForce * moveMult
        forceList.append(boundingBoxForce)



        # Pull towards Z position
        targHeightForce = pullTowardsTargetHeights(path, targetHeights[:path.shape[1]], 0.9*scaleFactA+0.1, 15)
        maxTargHeightForce = pullTowardsTargetHeights(path, targetHeights[:path.shape[1]], 0.8, 10) # Pull beginning and end harder
        heightForceMix = np.interp(
            np.arange(path.shape[1]),
            [LOCKED_PT_CNT, LOCKED_PT_CNT*2, POINT_COUNT-LOCKED_PT_CNT*2, POINT_COUNT-LOCKED_PT_CNT],
            [1.0, 0.0, 0.0, 1.0]
        )
        finalTargHeightForce = targHeightForce*(1.0-heightForceMix) + maxTargHeightForce*heightForceMix
        path += finalTargHeightForce*moveMult
        forceList.append(finalTargHeightForce)



        # Normalize differences between path norms
        pathNormSumForce = np.zeros_like(path)
        if scaleFactB > 0:
            addToPathAndSums(
                scaleFactB*normalizePathDists(path,  PT_SPACING, 0.5+0.5*scaleFactB, maxForce=5.0),
                path, pathNormSumForce, moveMult
            )
        addToPathAndSums(
            normalizePathDists(path,  PT_SPACING*2, 0.5, maxForce=5.0, pointOffset=2, dropZ=False),
            path, pathNormSumForce, moveMult
        )
        addToPathAndSums(
            normalizePathDists(path,  PT_SPACING*2, 0.5, maxForce=5.0, pointOffset=3, dropZ=False),
            path, pathNormSumForce, moveMult
        )
        forceList.append(pathNormSumForce)



        # Repel away from own path
        noSelfIntersectionForce = np.zeros_like(path)
        if scaleFactA > 0:
            noSelfIntersectionForce_broad = repelPathFromSelf(path, 4, 0.02, 40) * scaleFactA
            addToPathAndSums(noSelfIntersectionForce_broad, path, noSelfIntersectionForce, moveMult)
        if scaleFactB > 0:
            noSelfIntersectionForce_req = repelPathFromSelf(path, 2, 2.0, ABSOLUTE_MIN_PT_DIST*2.0) * scaleFactB
            noSelfIntersectionForce_req[:2] /= 4
            addToPathAndSums(noSelfIntersectionForce_req, path, noSelfIntersectionForce, moveMult)
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


        # Move to set points
        setPtForce = np.zeros_like(path)
        setPointIndices = setPointIndexList[pathIdx]
        setPoints = setPointList[pathIdx]
        setPtForce[:, setPointIndices] = setPoints[3] * (setPoints[:3] - path[:, setPointIndices])
        path += setPtForce * moveMult
        forceList.append(setPtForce)



        # Limit path curvature
        # Calculate turnaround points to adjust curvature
        radii, centerVectPrev, normCenterVect, centerVectNext, curvatureSign = approximatePathCurvatureXY(path, offset=1, includeCurvatureDir=True)
        curvatureFlipMag = np.zeros(path.shape[1])
        curvatureFlipMag[1:-1] = np.interp(distance_to_sign_change(curvatureSign), [1, 3], [1.0, 0.0])

        # Calculate slope inflection points to adjust curvature
        pathSlopes = calcPathSlope(path)
        slopeChangeMagnitudes = np.where(np.abs(np.diff(pathSlopes)) > np.abs(np.average(pathSlopes)), -1, 1)
        slopeSharpnessMag = np.zeros(path.shape[1])
        slopeSharpnessMag[1:-1] = np.interp(distance_to_sign_change(slopeChangeMagnitudes), [1, 2], [1.0, 0.0])
        
        curvAdjustMag = np.max([curvatureFlipMag, slopeSharpnessMag*0.4], axis=0)

        curvAdjustMag *= scaleFactB
        
        pathAngleForceSum = np.zeros_like(path)
        # if scaleFactB > 0: addToPathAndSums(scaleFactB*update_path_curvature(path, 25.0, 1e6, 0.1, 1.5, offset=1), path, pathAngleForceSum, moveMult)
        # addToPathAndSums(update_path_curvature(path, 25+30*curvAdjustMag, 40+50*curvAdjustMag+1e6*scaleFactC, 0.05, 1.5, offset=2), path, pathAngleForceSum, moveMult)
        # addToPathAndSums(update_path_curvature(path, 25+30*curvAdjustMag, 40+50*curvAdjustMag+1e6*scaleFactC, 0.05, 1.5, offset=3), path, pathAngleForceSum, moveMult)
        # if scaleFactA > 0: addToPathAndSums(scaleFactA*update_path_curvature(path, 30.0, 1e6, 0.02, 3.0, offset=4), path, pathAngleForceSum, moveMult)

        # if scaleFactB > 0:
        #     # basePathAngleForce = scaleFactB*update_path_curvature(path, 30.0, 1e6, 0.2, 10.0, offset=2)
        #     basePathAngleForce = scaleFactB*correctPathAngle(path, 3.0, 3.1, 1.0+3.0*scaleFactC, diffPointOffsetCnt=1)
        #     basePathAngleForce[:, LOCKED_PT_CNT:-LOCKED_PT_CNT] = 0.0
        #     addToPathAndSums(basePathAngleForce, path, pathAngleForceSum, moveMult)


        addToPathAndSums(scaleFactB*update_path_curvature(path, 50.0, 1e6, 0.1, 1.5, offset=1), path, pathAngleForceSum, moveMult)
        addToPathAndSums(scaleFactB*update_path_curvature(path, (40+30*curvAdjustMag)*scaleFactC, 50+50*curvAdjustMag+50*scaleFactD, 0.1, 1.5, offset=2), path, pathAngleForceSum, moveMult)
        addToPathAndSums(scaleFactB*update_path_curvature(path, (30+30*curvAdjustMag)*scaleFactC, 40+50*curvAdjustMag+50*scaleFactD, 0.1, 1.5, offset=3), path, pathAngleForceSum, moveMult)
        # addToPathAndSums(scaleFactA*update_path_curvature(path, 30.0+1e4*scaleFactD, 1e6, 0.02, 3.0, offset=4), path, pathAngleForceSum, moveMult)


        forceList.append(pathAngleForceSum)



        # Repel away from other paths
        repelForce = np.zeros_like(path)
        for cmpIdx in range(len(pathList)):
            if pathIdx == cmpIdx: continue
            
            if scaleFactB > 0:
                absoluteMinPathForce = scaleFactB * repelPoints(path, pathList[cmpIdx], 5.0, ABSOLUTE_MIN_PT_DIST*2.0) # Absolute required distance between points, only inpacts Z
                absoluteMinPathForce[:2] /= 1 + 1*scaleFactB
                addToPathAndSums(absoluteMinPathForce, path, repelForce, moveMult)

            if scaleFactA > 0:
                addToPathAndSums(
                    scaleFactA * repelPoints(path, pathList[cmpIdx], 0.01, 20+40*scaleFactA), # Broadly avoid other paths
                    path, repelForce, moveMult
                )

            if scaleFactB > 0:
                addToPathAndSums(
                    scaleFactB * repelPoints(path, pathList[cmpIdx][:, [0, -1]], 2.0, 40), # Avoid end points of other paths
                    path, repelForce, moveMult
                )

        if scaleFactA > 0:
            addToPathAndSums(
                scaleFactA * repelPoints(path, path[:, [0, -1]], 5.0, 60), # Avoid end points of own paths at first
                path, repelForce, moveMult
            )

        # Repel away from center lift
        addToPathAndSums(
            repelPoints(path, centerPoints, 4.0, 20+SCREW_RAD),
            path, repelForce, moveMult
        )
        forceList.append(repelForce)



        # Correct irregular slopes
        changeInSlopeForce = np.zeros_like(path)
        if scaleFactB > 0:
            addToPathAndSums(
                correctSlopeChange(path, 0.1*scaleFactB, 0.0*scaleFactB, upwardsForceMag=0.3*scaleFactB),
                path, changeInSlopeForce, moveMult
            )
            addToPathAndSums(
                correctSlopeChange(path, 0.1*scaleFactB, 0.0*scaleFactB, upwardsForceMag=0.01*scaleFactB, offset=2),
                path, changeInSlopeForce, moveMult
            )
            addToPathAndSums(correctSlopeChange(path, 0.1*scaleFactB, 0.0*scaleFactB, upwardsForceMag=0.2, offset=3), path, changeInSlopeForce, moveMult)
            addToPathAndSums(correctSlopeChange(path, 0.1*scaleFactB, 0.0*scaleFactB, upwardsForceMag=0.2, offset=4), path, changeInSlopeForce, moveMult)
            addToPathAndSums(correctSlopeChange(path, 0.1*scaleFactB, 0.0*scaleFactB, upwardsForceMag=0.2, offset=5), path, changeInSlopeForce, moveMult)
        forceList.append(changeInSlopeForce)



        # Do not slope up ever
        downHillForce = scaleFactB*preventUphillMotion(path, 1.0)
        path += downHillForce # Always apply at full force
        forceList.append(downHillForce)

        # if pathIdx == 4:
        #     plt.plot(downHillForce[2])
        #     plt.show()


        # Move to set points
        setPtForce = np.zeros_like(path)
        setPointIndices = setPointIndexList[pathIdx]
        setPoints = setPointList[pathIdx]
        setPtForce[:, setPointIndices] = setPoints[3] * (setPoints[:3] - path[:, setPointIndices])
        path += setPtForce * moveMult
        forceList.append(setPtForce)

        forcePointIndices = np.where(setPoints[3] == 1.0)[0]
        path[:, setPointIndices[forcePointIndices]] = setPoints[:3, forcePointIndices]



        # End of path updates

        # Calculate force data
        forceMags = [magnitude(fooFrc) for fooFrc in forceList]

        # Update dynamic temperature
        indicatorChar = '~'

        forceSums = np.sum(forceMags[:-1], axis=0) # Sum forces excluding setpointforce
        maxForceIdx = np.argmax(forceSums)

        # Calculate current temperature at point
        maxSumForce = forceSums[maxForceIdx]
        temperatureVal = np.interp(maxSumForce, PATH_RANDOMIZATION_FUNC[0], PATH_RANDOMIZATION_FUNC[1])
        temperatureDrop = np.interp(pathTempList[pathIdx], PATH_RANDOMIZATION_FUNC[1], PATH_RANDOMIZATION_FUNC[2])
        pathTempList[pathIdx] -= temperatureDrop

        if pathTempList[pathIdx] < -10:
            pathTempList[pathIdx] = -10
        
        # Update temperature if exceeds current
        if temperatureVal > pathTempList[pathIdx] and temperatureVal > -0.25:
            if temperatureVal - pathTempList[pathIdx] > 1.0:
                pathTempList[pathIdx] += 1
                indicatorChar = '+'
            else:
                pathTempList[pathIdx] = temperatureVal
                indicatorChar = '^'


        # Reset if failed to decrease
        temperatureIdx = pathIteration%TEMPERATURE_HISTORY_LEN
        prevTemperature = pathTempHistList[pathIdx][temperatureIdx]
        
        if maxSumForce > np.max(pathTempHistList[pathIdx]) + np.std(pathTempHistList[pathIdx]) and maxSumForce > -0.5:
            indicatorChar = '#'
            for tmpPathIdx in range(PATH_COUNT):
                pathTempList[tmpPathIdx] += TEMPERATURE_FAILURE_BOOST
                pathTempHistList[tmpPathIdx] = 1e9

        pathTempHistList[pathIdx][temperatureIdx] = maxSumForce

        # print(f"{indicatorChar} {maxSumForce:>8.5f} {pathTempList[pathIdx]:>8.5f} {scaleFactB:>1.1f} {scaleFactC:>1.1f}", end=' | ')
        print(f"{indicatorChar} {maxSumForce:<6.3f} {pathTempList[pathIdx]:<6.3f}", end=' | ')
            



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

        # Plot forces for last map
        if pathIdx%4 == 0:
            if REALTIME_PLOTTING_FORCEMAGS:
                medianForceMags = [np.max(fooMag) for fooMag in forceMags]
                medianForceMagQueue.put({
                    'boundingBoxForce': np.max(np.linalg.norm(boundingBoxForce, axis=0)),
                    'targHeightForce': np.max(np.linalg.norm(targHeightForce, axis=0)),
                    'pathNormForce': np.max(np.linalg.norm(pathNormSumForce, axis=0)),
                    'noSelfIntersectionForce': np.max(np.linalg.norm(noSelfIntersectionForce, axis=0)), 
                    'pathAngleForce': np.max(np.linalg.norm(pathAngleForceSum, axis=0)),
                    'repelForce': np.max(np.linalg.norm(repelForce, axis=0)),
                    'changeInSlopeForce': np.max(np.linalg.norm(changeInSlopeForce, axis=0)),
                    'downHillForce': np.max(np.linalg.norm(downHillForce, axis=0)),
                    'setPtForce': np.max(np.linalg.norm(setPtForce, axis=0)),
                })

        # Make all paths equal if requested
        if MIRROR_PATHS:
            for mirrorIdx in range(0, PATH_COUNT):
                if mirrorIdx == pathIdx: 
                    continue
                angle = getPathAnchorAngle(mirrorIdx) - getPathAnchorAngle(pathIdx)
                centeredPath = deepcopy(pathList[0])
                centeredPath[0] -= SIZE_X/2
                centeredPath[1] -= SIZE_Y/2
                rotPath = np.array(pf.doRotationMatrixes(centeredPath, [0, 0, angle]))
                rotPath[0] += SIZE_X/2
                rotPath[1] += SIZE_Y/2
                pathList[mirrorIdx] = rotPath

    # Print endline for logging
    print(' ')

    # if pathIteration%0 == 0:
    if True:
        # Use spline interpolation for additonal points
        # fullPaths = [subdividePath(path) for path in pathList]
        # fullPaths = [path for path in pathList]

        pkl.dump(pathList, open(WORKING_DIR+'path.pkl', 'wb'))
        pkl.dump(pathList, open(WORKING_DIR+f"PathDump/{str(pathDumpIter).rjust(6, '0')}.pkl", 'wb'))
        pathDumpIter += 1

pkl.dump(pathList, open(WORKING_DIR+'path.pkl', 'wb'))

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
