#! /usr/bin/env python

from solid2.extensions.bosl2 import circle, cuboid, sphere, cylinder, \
									heightfield, diff, tag, attach, \
									TOP, BOTTOM, CTR, metric_screws, rect, glued_circles, \
									chain_hull, conv_hull, hull, cube, union, trapezoid, teardrop, skin, sweep
from solid2.core import linear_extrude

import numpy as np
from copy import deepcopy
import pickle as pkl

from matplotlib import pyplot as plt
import sys
import os

from defs import *
from shared import *
from openScadGenerators import *

# Init test dir
os.makedirs(WORKING_DIR+'test/', exist_ok=True)

# Save screw base
outputAssembly = generateCenterScrewRotatingPart()
outputAssembly.save_as_scad(WORKING_DIR + "Screw.scad")

outputAssembly = cylinder(90, 5, 2)
outputAssembly.save_as_scad(WORKING_DIR + "Foot.scad")


# Load path data
pathList = pkl.load(open(WORKING_DIR+'path.pkl', 'rb'))
rotList = [subdividePath(path) for path in pathList]
rotList = [calculatePathRotations(path) for path in pathList]

outputAssembly = sphere(0)

# # Generate actual path geometry
for path, rot in zip(pathList, rotList):
	# outputAssembly += generateTrackFromPath(path, rot)
	# outputAssembly += generateTrackFromPath(path[:, :], rot[:, :])
	outputAssembly += generateTrackFromPathSubdiv(path[:, :], rot[:, :])

(outputAssembly).save_as_scad(WORKING_DIR + "test/tracks.scad")

# Add path sections to support screw lift
screwLoadAssembly = sphere(0)
screwLoadSupportAnchors = []
for pathIdx in range(PATH_COUNT):
	angle = getPathAnchorAngle(pathIdx)
	geometry, supportAnchors = generateScrewPathJoins(angle)
	screwLoadAssembly += geometry
	screwLoadSupportAnchors.append(supportAnchors+SCREW_POS[:, None])
screwLoadAssembly = screwLoadAssembly.translate(SCREW_POS)


# Get list of all points which require support
supportAnchors = [calculateSupportAnchorsForPath(path, rot) for path, rot in zip(pathList, rotList)]

supportPoints = np.concatenate(supportAnchors+screwLoadSupportAnchors, axis=1)

# Get list of all no-go points
noGoPoints = np.concatenate([path for path in pathList], axis=1) # Do not subdivide
# noGoPoints = np.concatenate([subdividePath(path) for path in pathList], axis=1) # Subdivide to get intermediate points
noGoPoints[2] -= MARBLE_RAD*2.0 # Repel away only upwards at 45 degree angle

# supportAnchorPointsConcat = np.concatenate([anchors for anchors in supportAnchors], axis=1) # get concat supportAnchors

# Calculate lift exclusion zone
liftNoGoRad = SCREW_RAD
liftNoGoZ = np.arange(BASE_OF_MODEL, SIZE_Z, 0.55555)
centerPoints = np.array([liftNoGoZ, liftNoGoZ, liftNoGoZ])
centerPoints[0, :] = SIZE_X/2 + liftNoGoRad*np.cos(liftNoGoZ)
centerPoints[1, :] = SIZE_Y/2 + liftNoGoRad*np.sin(liftNoGoZ)

liftNoGoPtCnt = 20
liftTrackNogo = np.zeros((3, liftNoGoPtCnt*PATH_COUNT))
for pathIdx in range(PATH_COUNT):
	angle = getPathAnchorAngle(pathIdx)
	liftTrackNogo[0, liftNoGoPtCnt*(pathIdx):liftNoGoPtCnt*(pathIdx+1)] = np.cos(angle)*(SCREW_RAD + MARBLE_RAD) + SCREW_POS[0]
	liftTrackNogo[1, liftNoGoPtCnt*(pathIdx):liftNoGoPtCnt*(pathIdx+1)] = np.sin(angle)*(SCREW_RAD + MARBLE_RAD) + SCREW_POS[1]
	liftTrackNogo[2, liftNoGoPtCnt*(pathIdx):liftNoGoPtCnt*(pathIdx+1)] = np.linspace(0, SIZE_Z, liftNoGoPtCnt)

noGoPoints = np.concatenate([noGoPoints, centerPoints, liftTrackNogo], axis=1)

# Add path of marble in 3d to check for intersections
marblePathGeometry = sphere(0)
for fooPath in pathList: marblePathGeometry += getShapePathSet(fooPath, np.zeros_like(fooPath), sphere(MARBLE_RAD))
if False:
	outputAssembly += marblePathGeometry

# Show screw and marble for example
#.translateZ(-(MARBLE_RAD+TRACK_RAD) + BASE_OF_MODEL)
rotatingScrew = generateCenterScrewRotatingPart()
# rotatingScrew += sphere(MARBLE_RAD, _fn=40).translateX(SCREW_RAD)
rotatingScrew = rotatingScrew.translate(SCREW_POS)


(screwLoadAssembly + outputAssembly).save_as_scad(WORKING_DIR + "test/FullPath.scad")

# Generate supports
if GENERATE_SUPPORTS:
	visPath = None
	if SUPPORT_VIS: visPath=WORKING_DIR+'vis/'
	supportColumns = calculateSupports(supportPoints, noGoPoints, visPath)
	supportGeometry = generateSupports(supportColumns)
else:
	supportGeometry = sphere(0)

	supportGeometry = getShapePathSet(supportPoints, None, sphere(1.5), returnIndividual=True)


# Generate base plate connections

# Visualize all support and no-go points
if False:
	for pt in np.swapaxes(supportPoints, 0, 1):
		outputAssembly += sphere(2).translate(pt)

# Show no go points
noGoDisplay = sphere(0)
for fooPt in np.swapaxes(noGoPoints, 0, 1):
	noGoDisplay += sphere(0.4).translate(fooPt)
(noGoDisplay).save_as_scad(WORKING_DIR + "test/noGo.scad")

(screwLoadAssembly + outputAssembly + supportGeometry).save_as_scad(WORKING_DIR + "MarbleRun.scad")

(screwLoadAssembly + outputAssembly + supportGeometry + rotatingScrew).save_as_scad(WORKING_DIR + "test/AllComponentsTogether.scad")
(((supportGeometry) & marblePathGeometry)).save_as_scad(WORKING_DIR + "test/supportIntersection.scad")
(supportGeometry).save_as_scad(WORKING_DIR + "test/supports.scad")
(screwLoadAssembly + rotatingScrew).save_as_scad(WORKING_DIR + "test/screwLoadAssembly.scad")

