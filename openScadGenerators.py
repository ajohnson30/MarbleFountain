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
import sys

from defs import *

def getShapePathSet(path, rotations, profile, returnIndividual=False, returnFunc=chain_hull):
	outProfiles = []
	if type(profile) != list: 
		profile = [profile]

	pathPts = np.swapaxes(path, 0, 1)
	if type(rotations) == np.ndarray:
		rotPts = np.swapaxes(rotations, 0, 1)

	for ii in range(len(pathPts)):
		currPt = pathPts[ii]
		fooProfile = profile[ii%len(profile)]

		if type(rotations) == np.ndarray:
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

def generateScrewSupports(inputPath, railSphere):
	outSupports = sphere(0)
	for idx in range(int(np.ceil(inputPath.shape[1]/SCREW_SUPPORT_GROUPING))):
		points = inputPath[:, idx*SCREW_SUPPORT_GROUPING:(idx+1)*SCREW_SUPPORT_GROUPING]
		
		# Move towards center
		joinPoint = np.average(points, axis=1)
		joinPoint[:2] /= 2
		joinPoint[2] -= SCREW_RAD/2

		if joinPoint[2] > SCREW_RAD/2:
			# Join midPoint to center
			centerPoint = deepcopy(joinPoint)
			centerPoint[:2] = 0
			centerPoint[2] -= SCREW_RAD/2

			outSupports += conv_hull()(*[
				railSphere.translate(centerPoint),
				railSphere.translate(joinPoint)
			])
		else:
			joinPoint[2] = 0
			outSupports += conv_hull()(*[
				railSphere.translate([0,0,0]),
				railSphere.translate(joinPoint)
			])

		for ptIdx in range(points.shape[1]):
			outSupports += conv_hull()(*[
				railSphere.translate(joinPoint),
				railSphere.translate(points[:, ptIdx])
			])
			

		# 	outSupports += railSphere.translate(points[:, ptIdx])
		# outSupports += railSphere.translate(centerPoint)

	return(outSupports)

def generateCenterScrewRotatingPart():
	# Define base objects
	railSphere = sphere(TRACK_RAD, _fn=UNIVERSAL_FN)
	outputScrew = sphere(0)

	# Calculate constants
	netRad = MARBLE_RAD+TRACK_RAD
	zOffsetOfSupportingRail = -np.sqrt(np.square(netRad) - np.square(SCREW_OUTER_TRACK_DIST))

	# return(innerRail + outerRail + linear_extrude(0.01)(circle(MARBLE_RAD, _fn=10)).rotate([90, 0, 90])) # Display profile

	# Generate height and angle of path at all points
	zPos = np.arange(0, SIZE_Z, SCREW_PITCH/SCREW_RESOLUTION)
	angle = zPos/SCREW_PITCH*2*np.pi
	zPos += np.sin(zPos*0.7)*1 # Subtely vary Z height to add interest
	zPos -= zOffsetOfSupportingRail # Lift all points slightly

	basePath = np.zeros((3, zPos.shape[0]))
	basePath[0] = np.cos(angle)
	basePath[1] = np.sin(angle)
	basePath[2] = zPos

	# Bottom rail
	bottomRailPath = deepcopy(basePath)
	bottomRailPath[:2] *= SCREW_RAD + SCREW_OUTER_TRACK_DIST
	bottomRailPath[2] += zOffsetOfSupportingRail
	outputScrew += getShapePathSet(bottomRailPath, None, railSphere)

	# Base rail
	baseRailPath = deepcopy(bottomRailPath[:, SCREW_RESOLUTION:(SCREW_RESOLUTION*2 +1)])
	baseRailPath[2] = 0.00000001
	outputScrew += getShapePathSet(baseRailPath, None, railSphere)
	# baseRailPath = deepcopy(bottomRailPath[:, 14:SCREW_RESOLUTION+1]) # If I do this all in 1 go it dies for some reason
	# baseRailPath[2] = 0.00000001
	# outputScrew += getShapePathSet(baseRailPath, None, railSphere)

	# Inside rail
	insideRailPath = deepcopy(basePath)
	insideRailPath[:2] *= SCREW_RAD - MARBLE_RAD - TRACK_RAD
	insideRailPath[:2, -SCREW_TOP_PUSH_PTS:] = ((SCREW_RAD - MARBLE_RAD - TRACK_RAD) + (np.linspace(0, MARBLE_RAD+SCREW_OUTER_TRACK_DIST+TRACK_RAD, SCREW_TOP_PUSH_PTS))) * (basePath[:2, -SCREW_TOP_PUSH_PTS:]) # Gradually push out marble
	# Gradually decrease height of top points
	insideRailPath[2, -SCREW_TOP_PUSH_PTS:] += np.linspace(0, zOffsetOfSupportingRail, SCREW_TOP_PUSH_PTS) * 0.6
	insideRailPath[2, -int(SCREW_TOP_PUSH_PTS/2):] += np.linspace(0, zOffsetOfSupportingRail, int(SCREW_TOP_PUSH_PTS/2)) * 0.4
	outputScrew += getShapePathSet(insideRailPath, None, railSphere)
		
	# Base inside rail
	baseInsideRailPath = deepcopy(basePath[:, :SCREW_RESOLUTION+1])
	baseInsideRailRads = np.ones(baseInsideRailPath.shape[1])
	baseInsideRailRads *= SCREW_RAD
	# Decrease part of path rad to allow ball entry
	decreaseRadCnt = int(np.floor(SCREW_RESOLUTION/3))
	decreaseRadMags = np.linspace(0, MARBLE_RAD+TRACK_RAD, decreaseRadCnt)
	baseInsideRailRads[-decreaseRadCnt:] -= decreaseRadMags
	# baseInsideRailRads[:decreaseRadCnt] -= MARBLE_RAD+TRACK_RAD
	# baseInsideRailRads[decreaseRadCnt:decreaseRadCnt*2] -= np.flip(decreaseRadMags)
	baseInsideRailPath[:2] *= baseInsideRailRads
	baseInsideRailPath[2] = -zOffsetOfSupportingRail # Set all Z to starting pos

	# Start at intersection of bottom rail
	intersectionIdx = np.where(bottomRailPath[2] > -zOffsetOfSupportingRail)[0][0] # Find intersection
	baseInsideRailPath = baseInsideRailPath[:, intersectionIdx:] # Truncate rail
	baseInsideRailPath[:, 0] = bottomRailPath[:, intersectionIdx] # Set starting point to intersection
	outputScrew += getShapePathSet(baseInsideRailPath, None, railSphere)

	# Supports
	outputScrew += generateScrewSupports(bottomRailPath, railSphere)
	outputScrew += generateScrewSupports(baseRailPath, railSphere)
	outputScrew += generateScrewSupports(insideRailPath, railSphere)
	outputScrew += generateScrewSupports(baseInsideRailPath, railSphere)

	# Add center shaft
	maxSupportHeight = np.max(insideRailPath[2]) - SCREW_RAD
	outputScrew += cylinder(maxSupportHeight+TRACK_RAD, TRACK_RAD*2, TRACK_RAD*0.95, _fn=UNIVERSAL_FN).translateZ(-TRACK_RAD)

	# MarblePath for viz
	if False:
		marblePath = deepcopy(basePath)
		marblePath[:2] *= SCREW_RAD
		marblePath[:2, -SCREW_TOP_PUSH_PTS:] = ((SCREW_RAD) + (np.linspace(0, MARBLE_RAD+SCREW_OUTER_TRACK_DIST+TRACK_RAD, SCREW_TOP_PUSH_PTS))) * (basePath[:2, -SCREW_TOP_PUSH_PTS:])
		outputScrew += getShapePathSet(marblePath, None, sphere(MARBLE_RAD, _fn=UNIVERSAL_FN))

	return(outputScrew)

def generateTrackFromPath(path, rotations):
	lowerDist = TRACK_SUPPORT_RAD*2
	trackToPathDist = MARBLE_RAD + TRACK_RAD
		
	# Calculate tall and short track profiles
	shortRail = linear_extrude(1)(circle(TRACK_SUPPORT_RAD, _fn=UNIVERSAL_FN)).rotate([90, 0, 90])

	tallRail =  chain_hull()(*[
		linear_extrude(0.2)(circle(TRACK_SUPPORT_RAD, _fn=UNIVERSAL_FN)).rotate([90, 0, 90]),
		linear_extrude(0.2)(circle(TRACK_SUPPORT_RAD, _fn=UNIVERSAL_FN).translate([0, -lowerDist])).rotate([90, 0, 90]),
	])

	rightTrackSet = [
		tallRail.translate([0, trackToPathDist*np.cos(TRACK_CONTACT_ANGLE), -trackToPathDist*np.sin(TRACK_CONTACT_ANGLE)]),
		shortRail.translate([0, trackToPathDist*np.cos(TRACK_CONTACT_ANGLE), -trackToPathDist*np.sin(TRACK_CONTACT_ANGLE)]),
	]

	leftTrackSet = [
		tallRail.translate([0, -trackToPathDist*np.cos(TRACK_CONTACT_ANGLE), -trackToPathDist*np.sin(TRACK_CONTACT_ANGLE)]),
		shortRail.translate([0, -trackToPathDist*np.cos(TRACK_CONTACT_ANGLE), -trackToPathDist*np.sin(TRACK_CONTACT_ANGLE)]),
	]

	tracks = sphere(0)
	for fooProfile in [rightTrackSet, leftTrackSet]:
		tracks += getShapePathSet(
			path,
			rotations,
			fooProfile 
			)
	return(tracks)



