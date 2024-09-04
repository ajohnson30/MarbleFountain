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
import subprocess as sp

from defs import *
from shared import *
import positionFuncs as pf

# Check if the directory exists
if not os.path.exists(WORKING_DIR):
    # Create the directory
    os.makedirs(WORKING_DIR)

# Check if the directory exists
if os.path.exists(WORKING_DIR+"PathDump/"): 
    sp.Popen(f"rm -r {WORKING_DIR}PathDump/", shell=True).wait()
os.makedirs(WORKING_DIR+"PathDump/")

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
    [0.0, 0.85, 1.0],
    [1.0, 0.1, 0.0]
) + startAndEndOffset


# Generate initial path
if 'SPIRAL' in sys.argv:
    spiralCnt = 3
    initAng = getPathAnchorAngle(0)
    # angles = np.interp(
    #     np.linspace(0.0, 1.0, targetHeights.shape[0]),
    #     [0.0, 0.05, 0.95, 1.0],
    #     [0.0, 0.0, np.pi*2*spiralCnt, np.pi*2*spiralCnt]
    # ) + initAng



    pathList = []
    for pathIdx in range(PATH_COUNT):
        pathAngle = getPathAnchorAngle(pathIdx)
        
        angles = np.interp(
            np.linspace(0.0, 1.0, targetHeights.shape[0]),
            [0.0, 0.1, 0.45, 0.55, 0.9, 1.0],
            [0.0, 0.3, np.pi*spiralCnt-0.4, np.pi*spiralCnt+0.4, np.pi*2*spiralCnt-0.3, np.pi*2*spiralCnt]
        ) + pathAngle


        angles = np.interp(
            np.linspace(0.0, 1.0, targetHeights.shape[0]),
            [0.0, 1.0],
            [0.0, np.pi*2*spiralCnt]
        ) + pathAngle

        # angles = np.interp(
        #     np.linspace(0.0, 1.0, targetHeights.shape[0]),
        #     [0.0, 0.1, 0.9, 1.0],
        #     [0.0, 0.0, np.pi*2*spiralCnt, np.pi*2*spiralCnt]
        # ) + pathAngle

        spiralRad = SCREW_RAD + PT_SPACING
        spiralRad = SIZE_Y/2
        path = np.zeros((3, targetHeights.shape[0]))
        path[0] = np.cos(angles)*SIZE_X/2
        path[1] = np.sin(angles)*SIZE_Y/2
        path[2] = targetHeights
        
        pathList.append(path)

    for pathIdx in range(PATH_COUNT):
        pathList[pathIdx][0] += SIZE_X/2 + (np.random.random(targetHeights.shape[0])-0.5)*2
        pathList[pathIdx][1] += SIZE_Y/2 + (np.random.random(targetHeights.shape[0])-0.5)*2


# Generate better starting points maybe
elif 'BETTER' in sys.argv:
    pathList = []

    for pathIdx in range(PATH_COUNT):
        path = np.zeros((3, POINT_COUNT), dtype=np.double)
        angle = getPathAnchorAngle(pathIdx)

        # Sort of random points
        for idx in range(3):
            if idx < 2:
                randPts = np.random.random(RANDOM_CNT+2)
                randPts *= BOUNDING_BOX[idx]
            else:
                randPts = np.linspace(SIZE_Z, 0, RANDOM_CNT+2)



            if idx < 2:
                if idx == 0:
                    startPos = np.cos(angle)*(SCREW_RAD + PT_SPACING) + BOUNDING_BOX[idx]/2
                    pos2 = np.cos(angle)*(SCREW_RAD + PT_SPACING*20) + BOUNDING_BOX[idx]/2
                elif idx == 1:
                    startPos = np.sin(angle)*(SCREW_RAD + PT_SPACING) + BOUNDING_BOX[idx]/2
                    pos2 = np.sin(angle)*(SCREW_RAD + PT_SPACING*20) + BOUNDING_BOX[idx]/2
                
                randPts[0] = startPos
                randPts[1] = pos2
                randPts[-1] = startPos
                randPts[-2] = pos2


            path[idx] = np.interp(np.linspace(0, RANDOM_CNT+1, POINT_COUNT), np.arange(RANDOM_CNT+2), randPts)
            path[idx] += 0.5 - np.random.random(len(path[idx]))
        
        # path[2, :] = targetHeights
        pathList.append(path)

# Generate slightly optimized starting points
elif 'SOLVE' in sys.argv:
    pathList = []

    for pathIdx in range(PATH_COUNT):
        path = np.zeros((3, POINT_COUNT), dtype=np.double)
        angle = getPathAnchorAngle(pathIdx)

        # Sort of random points
        randPath = np.zeros((3, RANDOM_CNT+2), dtype=np.double)
        for idx in range(3):
            if idx < 2:
                randPts = np.random.random(RANDOM_CNT+2)

                # Random points only in outside 1/4 of bounding box
                boxFrac = 1/4

                randPts *= BOUNDING_BOX[idx] * boxFrac
                randPts[np.where(randPts > BOUNDING_BOX[idx] * boxFrac/2)] += BOUNDING_BOX[idx] * boxFrac
                # Loop setting the start and end positions + preventing double crossings
                for ii in range(10):
                    # Do not cross center and back instantly
                    randPts[1:-1] = np.where(
                        (randPts[2:]-BOUNDING_BOX[idx]/2)*(randPts[:-2]-BOUNDING_BOX[idx]/2) < 0, 
                        randPts[1:-1],
                        np.random.random(RANDOM_CNT)*BOUNDING_BOX[idx], 
                    )

                    # Set first and last point
                    if idx == 0:
                        startPos = np.cos(angle)*(SCREW_RAD + PT_SPACING) + BOUNDING_BOX[idx]/2
                        pos2 = np.cos(angle)*(SCREW_RAD + PT_SPACING*20) + BOUNDING_BOX[idx]/2
                    elif idx == 1:
                        startPos = np.sin(angle)*(SCREW_RAD + PT_SPACING) + BOUNDING_BOX[idx]/2
                        pos2 = np.sin(angle)*(SCREW_RAD + PT_SPACING*20) + BOUNDING_BOX[idx]/2
                    
                    randPts[0] = startPos
                    randPts[1] = pos2
                    randPts[-1] = startPos
                    randPts[-2] = pos2

            # idx = 3, load set heights
            else:
                randPts = np.linspace(SIZE_Z, 0, RANDOM_CNT+2)


            randPath[idx] = np.array(randPts)
        
        targDist = 0.3*PT_SPACING * (POINT_COUNT / RANDOM_CNT)
        # targDist = 20
        print(f"targDist:{targDist}")
        SOLVE_ITERATIONS = 100
        for iter in range(SOLVE_ITERATIONS):
            boundingBoxForceCurve = [[-10, 0, 5, 10], [0.0, 0.1, 5, 30.0]]
            pathForce = normalizePathDists(randPath, targDist, iter/SOLVE_ITERATIONS, maxForce = 5.0, pointOffset = 1, dropZ = True)
            pathForce += (iter/SOLVE_ITERATIONS) * pushTowardsBoundingBox(randPath, BOUNDING_BOX, boundingBoxForceCurve, axCount=3)
            randPath[:, 1:-1] += pathForce[:, 1:-1]


        for idx in range(3):
            path[idx] = np.interp(np.linspace(0, RANDOM_CNT+1, POINT_COUNT), np.arange(RANDOM_CNT+2), randPath[idx])
            path[idx] += 0.5 - np.random.random(len(path[idx]))

        # path[2, :] = targetHeights
        pathList.append(path)
                


# Generate better starting points maybe
elif 'WIDE' in sys.argv:
    pathList = []

    for pathIdx in range(PATH_COUNT):
        path = np.zeros((3, POINT_COUNT), dtype=np.double)
        angle = getPathAnchorAngle(pathIdx)

        # Sort of random points
        for idx in range(3):
            if idx < 2:
                randPts = np.random.random(RANDOM_CNT+2)

                # Random points only in outside 1/4 of bounding box
                boxFrac = 1/4

                randPts *= BOUNDING_BOX[idx] * boxFrac
                randPts[np.where(randPts > BOUNDING_BOX[idx] * boxFrac/2)] += BOUNDING_BOX[idx] * boxFrac
                # Loop setting the start and end positions + preventing double crossings
                for ii in range(10):
                    # Do not cross center and back instantly
                    randPts[1:-1] = np.where(
                        (randPts[2:]-BOUNDING_BOX[idx]/2)*(randPts[:-2]-BOUNDING_BOX[idx]/2) < 0, 
                        randPts[1:-1],
                        np.random.random(RANDOM_CNT)*BOUNDING_BOX[idx], 
                    )

                    # Set first and last point
                    if idx == 0:
                        startPos = np.cos(angle)*(SCREW_RAD + PT_SPACING) + BOUNDING_BOX[idx]/2
                        pos2 = np.cos(angle)*(SCREW_RAD + PT_SPACING*20) + BOUNDING_BOX[idx]/2
                    elif idx == 1:
                        startPos = np.sin(angle)*(SCREW_RAD + PT_SPACING) + BOUNDING_BOX[idx]/2
                        pos2 = np.sin(angle)*(SCREW_RAD + PT_SPACING*20) + BOUNDING_BOX[idx]/2
                    
                    randPts[0] = startPos
                    randPts[1] = pos2
                    randPts[-1] = startPos
                    randPts[-2] = pos2

            # idx = 3, load set heights
            else:
                randPts = np.linspace(SIZE_Z, 0, RANDOM_CNT+2)



            path[idx] = np.interp(np.linspace(0, RANDOM_CNT+1, POINT_COUNT), np.arange(RANDOM_CNT+2), randPts)
            path[idx] += 0.5 - np.random.random(len(path[idx]))
        
        # path[2, :] = targetHeights
        pathList.append(path)
                


else:
    # Default gen
    pathList = []
    for pathIdx in range(PATH_COUNT):
        path = randomPath(POINT_COUNT, BOUNDING_BOX, pathIdx)
        path[2, :] = targetHeights
        pathList.append(path)

pkl.dump(pathList, open(WORKING_DIR+'path.pkl', 'wb'))
pkl.dump(pathList, open(WORKING_DIR+'PathDump/000000.pkl', 'wb'))