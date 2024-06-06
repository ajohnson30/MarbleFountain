#! /usr/bin/env python

from solid2.extensions.bosl2 import circle, cuboid, sphere, cylinder, \
									heightfield, diff, tag, attach, \
									TOP, BOTTOM, CTR, metric_screws, rect, glued_circles, \
									chain_hull, conv_hull, hull, cube, union, trapezoid, teardrop, skin, sweep

from solid2.extensions.bosl2.turtle3d import turtle3d

from solid2.core import linear_extrude

import numpy as np
from scipy.interpolate import splprep, splev
from scipy.spatial.distance import cdist
from random import random
from copy import deepcopy
import pickle as pkl

from defs import *

def getShapePathSet(path, rotations, profile, returnIndividual=False, returnFunc=chain_hull):
	outProfiles = []
	if type(profile) != list: 
		profile = [profile]

	pathPts = np.swapaxes(path, 0, 1)
	if type(rotations) != None:
		rotPts = np.swapaxes(rotations, 0, 1)

	for ii in range(len(pathPts)):
		currPt = pathPts[ii]
		fooProfile = profile[ii%len(profile)]

		if type(rotations) != None:
			currRot = rotPts[ii]
			fooProfile = fooProfile.rotate([np.degrees(currRot[1]), 0, 0]) # Tilt
			fooProfile = fooProfile.rotate([0, 0, np.degrees(currRot[0])]) # Rotate about Z
		fooProfile = fooProfile.translate(currPt) # Move to final position
		outProfiles.append(fooProfile)
	
	if returnIndividual:
		output = sphere(0)
		for ii in range(len(outProfiles)-1):
			output += outProfiles[ii]
		return(output)

	return(returnFunc()(*outProfiles))

def getCenterScrewRotatingPart():

	# Make Screw rail
	netRad = MARBLE_RAD+TRACK_RAD
	zOffsetOfSupportingRail = -np.sqrt(np.square(netRad) - np.square(SCREW_OUTER_TRACK_DIST))
	innerRail = sphere(TRACK_RAD, _fn=20).translate([0, -netRad, 0])
	outerRail = sphere(TRACK_RAD, _fn=20).translate([0, SCREW_OUTER_TRACK_DIST, zOffsetOfSupportingRail])

	# return(innerRail + outerRail + linear_extrude(0.01)(circle(MARBLE_RAD, _fn=10)).rotate([90, 0, 90])) # Display profile

	zPos = np.arange(0, SIZE_Z, SCREW_PITCH/SCREW_RESOLUTION)
	angle = zPos/SCREW_PITCH*2*np.pi

	zPos += np.sin(zPos*0.7)*1

	print(angle)
	path = np.zeros((3, zPos.shape[0]))
	path[0] = SCREW_RAD*np.cos(angle)
	path[1] = SCREW_RAD*np.sin(angle)
	path[2] = zPos

	rotations = np.zeros((2, zPos.shape[0]))
	rotations[0] = angle + np.pi/2
	rotations[1] = np.arctan2(SCREW_PITCH/SCREW_RESOLUTION, 2*np.pi/SCREW_RESOLUTION)
	

	outputScrew = getShapePathSet(path, rotations, innerRail)
	outputScrew += getShapePathSet(path, rotations, outerRail)
	outputScrew += getShapePathSet(path, rotations, sphere(MARBLE_RAD, _fn=20))

	# Add base support
	basePath = deepcopy(path[:SCREW_RESOLUTION])
	basePath[2] = 0
	baseBottom = zOffsetOfSupportingRail-TRACK_RAD

	outputScrew += getShapePathSet(basePath, rotations, innerRail)
	outputScrew += getShapePathSet(basePath, rotations, outerRail)

	return(outputScrew)


