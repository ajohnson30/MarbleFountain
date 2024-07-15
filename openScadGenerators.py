#! /usr/bin/env python
from solid2.extensions.bosl2 import circle, cuboid, sphere, cylinder, \
									heightfield, diff, tag, attach, \
									TOP, BOTTOM, CTR, metric_screws, rect, glued_circles, \
									chain_hull, conv_hull, hull, cube, union, trapezoid, teardrop, skin, sweep, polygon

from solid2.extensions.bosl2.turtle3d import turtle3d

from solid2.core import linear_extrude

import numpy as np
from scipy.interpolate import splprep, splev
from scipy.spatial.distance import cdist
from random import random
from copy import deepcopy
import pickle as pkl
import sys
import os

from defs import *
from shared import *
import positionFuncs as pf

def generateCutoutForPrinting():
	CUTOUT_Z = 2.0
	baseCutout = linear_extrude(300)(trapezoid(CUTOUT_Z, 3.0, 1.0)).rotate([90, 0, 90]).translate([-150, 0, CUTOUT_Z/2])
	return(baseCutout + baseCutout.rotateZ(60) + baseCutout.rotateZ(120))

# Extrude shape along path with rotaions
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

# Screw generation
def generateScrewSupports(inputPath, railSphere):
	outSupports = sphere(0)
	# for idx in range(int(np.ceil(inputPath.shape[1]/SCREW_SUPPORT_GROUPING))):
	idx = 0
	joinPoints = []
	while True:
		groupSize = int(np.random.rand()*np.diff(SCREW_SUPPORT_GROUPING) + SCREW_SUPPORT_GROUPING[0])
		points = inputPath[:, idx:idx+groupSize]
		if points.shape[1] == 0:
			break

		idx += groupSize

		# Move towards center
		joinPoint = np.average(points, axis=1)
		joinPoint[:2] /= 2
		joinPoint[2] -= SCREW_RAD/2

		if joinPoint[2] > SCREW_RAD/2 + BASE_OF_MODEL + TRACK_RAD:
			# Join midPoint to center
			centerPoint = deepcopy(joinPoint)
			centerPoint[:2] = 0
			centerPoint[2] -= SCREW_RAD/2

			outSupports += conv_hull()(*[
				railSphere.translate(centerPoint),
				railSphere.translate(joinPoint)
			])
		else:
			joinPoint[2] = BASE_OF_MODEL + TRACK_RAD
			joinPoint[:2] = 0.0
			# outSupports += conv_hull()(*[
			# 	railSphere.translate([0,0,BASE_OF_MODEL + TRACK_RAD]),
			# 	railSphere.translate(joinPoint)
			# ])

		for ptIdx in range(points.shape[1]):
			outSupports += conv_hull()(*[
				railSphere.translate(joinPoint),
				railSphere.translate(points[:, ptIdx])
			])
		
		joinPoints.append(joinPoint)
		# 	outSupports += railSphere.translate(points[:, ptIdx])
		# outSupports += railSphere.translate(centerPoint)

	everyNth = 4
	joinPointArr = np.swapaxes(joinPoints, 0, 1)
	for connLoop in range(everyNth):
		outSupports += getShapePathSet(joinPointArr[:, connLoop::everyNth], None, railSphere)

	return(outSupports)

# Generate the actual rotating part of the screw lift
def generateCenterScrewRotatingPart():
	# Define base objects
	railSphere = sphere(TRACK_RAD, _fn=UNIVERSAL_FN)
	outputScrew = sphere(0)

	# Calculate constants
	netRad = MARBLE_RAD+TRACK_RAD
	zOffsetOfSupportingRail = -np.sqrt(np.square(netRad) - np.square(SCREW_OUTER_TRACK_DIST))

	# return(innerRail + outerRail + linear_extrude(0.01)(circle(MARBLE_RAD, _fn=10)).rotate([90, 0, 90])) # Display profile
	BASE_POS_DROP = MARBLE_RAD/2
	# Generate height and angle of path at all points
	zPos = np.arange(-BASE_POS_DROP, SIZE_Z+MARBLE_RAD*2.5, SCREW_PITCH/SCREW_RESOLUTION)
	angle = zPos/SCREW_PITCH*2*np.pi
	# zPos += np.sin(zPos*0.7)*1 # Subtely vary Z height to add interest
	# zPos -= zOffsetOfSupportingRail # Lift all points slightly

	basePath = np.zeros((3, zPos.shape[0]))
	basePath[0] = np.cos(angle)
	basePath[1] = np.sin(angle)
	basePath[2] = zPos


	# Bottom rail
	bottomRailPath = deepcopy(basePath)
	bottomRailPath[:2] *= SCREW_RAD + SCREW_OUTER_TRACK_DIST
	bottomRailPath[2] += zOffsetOfSupportingRail
	# bottomRailPath[2, bottomRailPath[2] > SIZE_Z-netRad*np.sin(TRACK_CONTACT_ANGLE)] = SIZE_Z-netRad*np.sin(TRACK_CONTACT_ANGLE)
	# bottomRailPath[2, -1] = SIZE_Z
	outputScrew += getShapePathSet(bottomRailPath, None, railSphere)

	# Base rail
	baseRailPath = deepcopy(bottomRailPath[:, :SCREW_RESOLUTION+1])
	baseRailPath[2] = zOffsetOfSupportingRail - BASE_POS_DROP
	outputScrew += getShapePathSet(baseRailPath, None, railSphere)
	# baseRailPath = deepcopy(bottomRailPath[:, 14:SCREW_RESOLUTION+1]) # If I do this all in 1 go it dies for some reason
	# baseRailPath[2] = 0.00000001
	# outputScrew += getShapePathSet(baseRailPath, None, railSphere)

	# Inside rail
	insideRailPath = deepcopy(basePath)
	insideRailPath[:2] *= SCREW_RAD - MARBLE_RAD - TRACK_RAD
	# Push marble out at top
	insideRailPath[:2, -SCREW_TOP_PUSH_PTS:] = (
		((SCREW_RAD - MARBLE_RAD - TRACK_RAD) 
   		+ (np.linspace(0, (MARBLE_RAD+SCREW_OUTER_TRACK_DIST+TRACK_RAD)/2, SCREW_TOP_PUSH_PTS))) * (basePath[:2, -SCREW_TOP_PUSH_PTS:])
	 )
	# Gradually decrease height of top points
	# insideRailPath[2, insideRailPath[2] > SIZE_Z] = SIZE_Z
	# insideRailPath[2, -SCREW_TOP_PUSH_PTS:] += np.linspace(0, zOffsetOfSupportingRail, SCREW_TOP_PUSH_PTS) * 0.6
	# insideRailPath[2, -int(SCREW_TOP_PUSH_PTS/2):] += np.linspace(0, zOffsetOfSupportingRail, int(SCREW_TOP_PUSH_PTS/2)) * 0.4
	outputScrew += getShapePathSet(insideRailPath, None, railSphere)
		
	# Base inside rail
	baseInsideRailPath = deepcopy(basePath[:, :SCREW_RESOLUTION+1])

	baseInsideRailPath[2] =  -BASE_POS_DROP # Set all Z to starting pos

	# baseInsideRailRads = np.ones(baseInsideRailPath.shape[1])
	# baseInsideRailRads *= SCREW_RAD
	
	# Start at intersection of bottom rail
	intersectionIdx = np.where(bottomRailPath[2] > 0.0)[0][0] # Find intersection
	baseInsideRailPath = baseInsideRailPath[:, intersectionIdx:] # Truncate rail

	# Gradually increase radius
	baseInsideRailRads = np.interp(
		np.linspace(0.0, 1.0, baseInsideRailPath.shape[1]),
		[0.0, 0.2, 0.4, 1.0],
		[SCREW_RAD, SCREW_RAD, SCREW_RAD-(MARBLE_RAD+TRACK_RAD), SCREW_RAD-(MARBLE_RAD+TRACK_RAD)]
	)
	baseInsideRailPath[:2, :] *= baseInsideRailRads

	baseInsideRailPath[:, 0] = bottomRailPath[:, intersectionIdx] # Set starting point to intersection


	outputScrew += getShapePathSet(baseInsideRailPath, None, railSphere)

	# Add center shaft
	maxSupportHeight = np.max(insideRailPath[2]) - SCREW_RAD
	outputScrewSupports = cylinder(SCREW_RAD*2, SCREW_RAD*0.8, TRACK_RAD*0.95, _fn=HIGHER_RES_FN).translateZ(BASE_OF_MODEL)
	outputScrewSupports += cylinder(maxSupportHeight-BASE_OF_MODEL, TRACK_RAD*2, TRACK_RAD*0.95, _fn=HIGHER_RES_FN).translateZ(BASE_OF_MODEL)


	# # Supports
	if GENERATE_SUPPORTS:
		outputScrewSupports += generateScrewSupports(bottomRailPath, railSphere)
		outputScrewSupports += generateScrewSupports(baseRailPath, railSphere)
		outputScrewSupports += generateScrewSupports(insideRailPath, railSphere)
		outputScrewSupports += generateScrewSupports(baseInsideRailPath, railSphere)

	# Motor Shaft Cutout
	if MOTOR_TYPE == 'SMALL_DC':
		outputScrewSupports -= (cylinder(12, 1.5, 1.5, _fn=HIGHER_RES_FN) & cube([10, 10, 8]).translate([1.5-2.4, -5, 0])).translateZ(BASE_OF_MODEL-2)
	elif MOTOR_TYPE == 'NEMA17':
		outputScrewSupports -= (cylinder(20.0, 2.55, 2.55, _fn=HIGHER_RES_FN)).translateZ(BASE_OF_MODEL-2)


	# MarblePath for viz
	if False:
		marblePath = deepcopy(basePath)
		marblePath[:2] *= SCREW_RAD
		marblePath[:2, -SCREW_TOP_PUSH_PTS:] = ((SCREW_RAD) + (np.linspace(0, MARBLE_RAD+SCREW_OUTER_TRACK_DIST+TRACK_RAD, SCREW_TOP_PUSH_PTS))) * (basePath[:2, -SCREW_TOP_PUSH_PTS:])
		outputScrew += getShapePathSet(marblePath, None, sphere(MARBLE_RAD, _fn=UNIVERSAL_FN))

	# Add vent holes for SLA printing
	ventHoles =  generateCutoutForPrinting().translateZ(BASE_OF_MODEL-0.1)

	return((outputScrewSupports + outputScrew) - ventHoles)

# Connect path to  screw
def generateScrewPathJoins(angle):
	# Define base objects
	railSphere = sphere(TRACK_RAD, _fn=UNIVERSAL_FN)
	outputGeometry = sphere(0)

	# Calculate constants
	netRad = MARBLE_RAD+TRACK_RAD
	zOffsetOfSupportingRail = -np.sqrt(np.square(netRad) - np.square(SCREW_OUTER_TRACK_DIST))

	vertRailDistFromSpiral = SCREW_OUTER_TRACK_DIST + SCREW_VERT_RAIL_MARGIN + TRACK_RAD*2
	vertRailSideOffset = np.sqrt(np.square(netRad) - np.square(vertRailDistFromSpiral))

	# vertRail = conv_hull()(railSphere, railSphere.translateZ(SIZE_Z))
	# outputGeometry += vertRail.translate([-vertRailDistFromSpiral, -vertRailSideOffset, zOffsetOfSupportingRail])
	# outputGeometry += vertRail.translate([-vertRailDistFromSpiral, +vertRailSideOffset, zOffsetOfSupportingRail])
	

	# leftRailPath = [[-vertRailDistFromSpiral, -vertRailSideOffset, zOffsetOfSupportingRail], [-vertRailDistFromSpiral, -vertRailSideOffset, SIZE_Z+zOffsetOfSupportingRail]]
	# rightRailPath = [[-vertRailDistFromSpiral, +vertRailSideOffset, zOffsetOfSupportingRail], [-vertRailDistFromSpiral, -vertRailSideOffset, SIZE_Z+zOffsetOfSupportingRail]]
	
	supportPoints = []

	PT_CNT = 5

	# Init default rail path
	railPath = np.zeros((3, 2+PT_CNT*2))
	railPath[0] = vertRailDistFromSpiral
	railPath[1] = vertRailSideOffset
	railPath[2] = zOffsetOfSupportingRail

	# First point matches tracks
	railPath[0, 0] = PT_SPACING*2
	railPath[1, 0] = netRad*np.cos(TRACK_CONTACT_ANGLE)
	railPath[2, 0] = -netRad*np.sin(TRACK_CONTACT_ANGLE) + INITIAL_POINT_MULT_SLOPE

	entryRad = netRad+TRACK_RAD/4

	# Lower part of semicircular feature
	bottomAngles = np.linspace(-TRACK_CONTACT_ANGLE, 0, PT_CNT)
	railPath[1, 1:PT_CNT+1] = entryRad*np.cos(bottomAngles)
	railPath[2, 1:PT_CNT+1] = entryRad*np.sin(bottomAngles)

	# Upper part of semicircular feature
	distMM = 5
	bottomAngles = np.linspace(0, 0, PT_CNT)
	railPath[1, PT_CNT+1:2*PT_CNT+1] = (np.cos(np.linspace(0, np.pi, PT_CNT))+1)/2 * (entryRad-vertRailSideOffset) + vertRailSideOffset
	railPath[2, PT_CNT+1:2*PT_CNT+1] = np.linspace(0, distMM, PT_CNT)

	# Last point at top 
	railPath[2, -1] = SIZE_Z + entryRad*np.sin(-TRACK_CONTACT_ANGLE) # Mate with bottom point on top loop

	# Save, mirror, save
	outputGeometry += getShapePathSet(railPath, None, railSphere)
	rightRail = deepcopy(railPath)
	rightRail[1] *= -1
	outputGeometry += getShapePathSet(rightRail, None, railSphere)

	# Add legs
	legPath = np.zeros((3, 4))
	legPath[:, 0] = railPath[:, 1] # Awful code but I'm tired and want to print this tomorrow
	legPath[:, 1] = railPath[:, 1]
	legPath[:, 2] = railPath[:, 1]
	legPath[:, 3] = railPath[:, 1]

	legPath[2, 1] = BASE_OF_MODEL
	legPath[:, 2] = railPath[:, 0]
	legPath[2, 2] -= TRACK_RAD*2


	# Save, mirror, save
	outputGeometry += getShapePathSet(legPath, None, railSphere)
	legPath = deepcopy(legPath)
	legPath[1] *= -1
	outputGeometry += getShapePathSet(legPath, None, railSphere)


	# Add top loop
	UPPER_PT_CNT = 10
	topRailPath = np.zeros((3, 1+UPPER_PT_CNT))
	topRailPath[0] = vertRailDistFromSpiral

	# First point matches tracks
	topRailPath[0, 0] = PT_SPACING*2
	topRailPath[1, 0] = netRad*np.cos(TRACK_CONTACT_ANGLE)
	topRailPath[2, 0] = SIZE_Z-netRad*np.sin(TRACK_CONTACT_ANGLE) - INITIAL_POINT_MULT_SLOPE
	
	# Circular feature at top
	angleSet = np.linspace(-np.arccos(vertRailSideOffset/entryRad), np.pi/2, UPPER_PT_CNT)
	topRailPath[1, 1:UPPER_PT_CNT+1] = entryRad*np.cos(angleSet)
	topRailPath[2, 1:UPPER_PT_CNT+1] = entryRad*np.sin(angleSet)
	topRailPath[2, 1:UPPER_PT_CNT+1] += 1.2*entryRad*np.interp(
		np.linspace(0.0, 1.0, UPPER_PT_CNT),
		[0.0, 0.1, 0.4, 1.0],
		[0.0, 0.0, 0.9, 1.0]
	)
	topRailPath[2, 1:UPPER_PT_CNT+1] += SIZE_Z

	# Save, mirror, save
	outputGeometry += getShapePathSet(topRailPath, None, railSphere)
	rightRail = deepcopy(topRailPath)
	rightRail[1] *= -1
	outputGeometry += getShapePathSet(rightRail, None, railSphere)

	# Add legs
	legPath = np.zeros((3, 3))
	legPath[:, 0] = topRailPath[:, 1]
	legPath[:, 1] = topRailPath[:, 0]
	legPath[2, 1] -= TRACK_SUPPORT_RAD*2
	legPath[:, 2] = topRailPath[:, 1]
	legPath[2, 2] -= MARBLE_RAD*2

	# Save, mirror, save
	outputGeometry += getShapePathSet(legPath, None, railSphere)
	legPath = deepcopy(legPath)
	legPath[1] *= -1
	outputGeometry += getShapePathSet(legPath, None, railSphere)

	# Add supports to vertical columns
	minZ = np.min(topRailPath[2])
	maxZ = np.max(railPath[2, :-1])
	SUPPORT_PTS = 19
	SUPPORT_DIST = TRACK_RAD*3
	supportPath = np.zeros((3, SUPPORT_PTS))
	supportPath[0, :] = railPath[0, -1]
	supportPath[1, :] = railPath[1, -1]
	supportPath[2, :] = np.linspace(minZ, maxZ, SUPPORT_PTS) # Interpolate Z positions
	supportPath[0, 1::2] += SUPPORT_DIST

	# Save, mirror, save
	outputGeometry += getShapePathSet(supportPath, None, railSphere)
	outputGeometry += getShapePathSet(supportPath[:, 1::2], None, railSphere)
	supportPath_right = deepcopy(supportPath)
	supportPath_right[1] *= -1
	outputGeometry += getShapePathSet(supportPath_right, None, railSphere)
	outputGeometry += getShapePathSet(supportPath_right[:, 1::2], None, railSphere)

	# Add supporting connections to adjacent path
	if CONNECT_LIFTS:
		supportBasePos = [vertRailDistFromSpiral+SUPPORT_DIST+SCREW_RAD, vertRailSideOffset, 0]
		supportMatchPos = pf.doRotationMatrixes([vertRailDistFromSpiral+SCREW_RAD+SUPPORT_DIST, -vertRailSideOffset, 0], [0, 0, 2*np.pi/PATH_COUNT])
		# supportMatchPos[0] -= SCREW_RAD

		supportBaseDist = np.linalg.norm(supportBasePos)
		supportBaseAngle = np.arctan2(supportBasePos[0], supportBasePos[1])
		supportMatchAngle = np.arctan2(supportMatchPos[0], supportMatchPos[1])
		
		supportPtCnt = LIFT_SUPPORT_CROSSES * LIFT_SUPPORT_SUBDIV
		supportPts = np.zeros((3, supportPtCnt), dtype=np.double)
		supportPts[2] = np.linspace(supportPath[2, 1], supportPath[2, -2], supportPtCnt)

		angleList = np.linspace(supportBaseAngle, supportMatchAngle, int((LIFT_SUPPORT_SUBDIV-1)/2))
		angleList = np.concatenate([angleList, np.flip(angleList[:-1])])
		for angleIdx in range(len(angleList)):
			fooAng = angleList[angleIdx]
			supportPts[0, angleIdx::len(angleList)] = np.cos(fooAng) * supportBaseDist
			supportPts[1, angleIdx::len(angleList)] = np.sin(fooAng) * supportBaseDist
		
		supportPts[0] -= SCREW_RAD

		outputGeometry += getShapePathSet(supportPts, None, railSphere)

		supportPts_right = deepcopy(supportPts)
		supportPts_right[1] *= -1
		outputGeometry += getShapePathSet(supportPts_right, None, railSphere)
		

	supportPoints = np.concatenate([supportPath[:, 1::2], supportPath_right[:, 1::2]], axis=1)
	supportPoints[0] += SCREW_RAD
	supportPoints = pf.doRotationMatrixes(supportPoints, [0, 0, angle])

	return(outputGeometry.translateX(SCREW_RAD).rotateZ(180.0*angle/np.pi), supportPoints)

# Generate track geometry
def generateTrackFromPath(path, rotations):
	lowerDist = TRACK_SUPPORT_RAD*2
	trackToPathDist = MARBLE_RAD + TRACK_RAD
		
	# Calculate tall and short track profiles
	shortRail = linear_extrude(1)(circle(TRACK_SUPPORT_RAD, _fn=UNIVERSAL_FN)).rotate([90, 0, 90])

	tallRail =  conv_hull()(*[
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

def generateTrackFromPathSubdiv(path, rotations):
	# Use spline interpolation for additonal points
	fullPath = path # subdividePath(path)
	fullRots = rotations # calculatePathRotations(fullPath)

	lowerDist = TRACK_SUPPORT_RAD*2
	trackToPathDist = MARBLE_RAD + TRACK_RAD
		
	# Calculate tall and short track profiles
	shortRail = linear_extrude(0.2)(circle(TRACK_SUPPORT_RAD, _fn=UNIVERSAL_FN)).rotate([90, 0, 90])

	# Generate circular profile
	angleList = np.linspace(0.0, 2*np.pi, UNIVERSAL_FN)
	railPoints = np.zeros((UNIVERSAL_FN, 2))
	railPoints[:, 0] = TRACK_SUPPORT_RAD*np.cos(angleList)
	railPoints[:, 1] = TRACK_SUPPORT_RAD*np.sin(angleList)

	tallPoints = deepcopy(railPoints)
	tallPoints[int(UNIVERSAL_FN/2):, 1] -= lowerDist
	tallRail = linear_extrude(0.2)(polygon(tallPoints)).rotate([90, 0, 90])

	medTallPoints = deepcopy(railPoints)
	medTallPoints[int(UNIVERSAL_FN/2):, 1] -= lowerDist*0.75
	medTallRail = linear_extrude(0.2)(polygon(medTallPoints)).rotate([90, 0, 90])

	medPoints = deepcopy(railPoints)
	medPoints[int(UNIVERSAL_FN/2):, 1] -= lowerDist*0.5
	medRail = linear_extrude(0.2)(polygon(medPoints)).rotate([90, 0, 90])
	# tallRail = polygon(outPts).rotate([90, 0, 90])

	medShortPoints = deepcopy(railPoints)
	medShortPoints[int(UNIVERSAL_FN/2):, 1] -= lowerDist*0.25
	medShortRail = linear_extrude(0.2)(polygon(medShortPoints)).rotate([90, 0, 90])

	rightSupportOffset = [0, trackToPathDist*np.cos(TRACK_CONTACT_ANGLE), -trackToPathDist*np.sin(TRACK_CONTACT_ANGLE)]
	leftSupportOffset = [0, -trackToPathDist*np.cos(TRACK_CONTACT_ANGLE), -trackToPathDist*np.sin(TRACK_CONTACT_ANGLE)]

	rightTrackSet = [
		tallRail.translate(rightSupportOffset),
		medTallRail.translate(rightSupportOffset),
		medRail.translate(rightSupportOffset),
		medShortRail.translate(rightSupportOffset),
		shortRail.translate(rightSupportOffset),
		medShortRail.translate(rightSupportOffset),
		medRail.translate(rightSupportOffset),
		medTallRail.translate(rightSupportOffset),
	]

	leftTrackSet = [
		tallRail.translate(leftSupportOffset),
		medTallRail.translate(leftSupportOffset),
		medRail.translate(leftSupportOffset),
		medShortRail.translate(leftSupportOffset),
		shortRail.translate(leftSupportOffset),
		medShortRail.translate(leftSupportOffset),
		medRail.translate(leftSupportOffset),
		medTallRail.translate(leftSupportOffset),
	]

	tracks = sphere(0)
	for fooProfile in [rightTrackSet, leftTrackSet]:
		tracks += getShapePathSet(
			fullPath,
			fullRots,
			fooProfile 
			)
	return(tracks)

def applyRotationsToPoint(points, rotation):
	# tiltedPoint = pf.doRotationMatrixes(points, [rotation[1], 0.0, 0.0])
	# return pf.doRotationMatrixes(tiltedPoint, [0.0, 0.0, rotation[0]])
	return pf.doRotationMatrixes(points, [rotation[1], 0, rotation[0]])

# Get the support anchors for the track
def calculateSupportAnchorsForPath(path, rotations):
	lowerDist = TRACK_SUPPORT_RAD*2
	trackToPathDist = MARBLE_RAD + TRACK_RAD

	# Calculate the offset of each of the supporting points relative to the frame
	rightSupportOffset = [0, trackToPathDist*np.cos(TRACK_CONTACT_ANGLE), -trackToPathDist*np.sin(TRACK_CONTACT_ANGLE)-lowerDist]
	leftSupportOffset = [0, -trackToPathDist*np.cos(TRACK_CONTACT_ANGLE), -trackToPathDist*np.sin(TRACK_CONTACT_ANGLE)-lowerDist]

	supportSectionsCount = int(np.ceil(path.shape[1]/4))
	anchorPts = np.zeros((3, supportSectionsCount*2), dtype=np.double)
	for ii in range(0, supportSectionsCount*2, 2):
		anchorPts[:, ii] = path[:, ii*2] + applyRotationsToPoint(rightSupportOffset, rotations[:, ii*2])
		anchorPts[:, ii+1] = path[:, ii*2] + applyRotationsToPoint(leftSupportOffset, rotations[:, ii*2])

	return(anchorPts)

# Data structure to simplify handling support columns
class column:
	def __init__(self, startPos, _size):
		self.currPos = np.array(startPos, dtype=np.double)
		self.prevPos = np.array(startPos, dtype=np.double)
		self.posHist = []
		self.sumAcc = np.zeros((2), dtype=np.double)
		self.velocity = np.zeros((2), dtype=np.double)
		self.size = _size
		self.mergedFrom = []
		self.mergingInto = -1
		self.mergedSize = -1

# Calculate repulsion away from avoidPts
def calculateRepulsivesSupportForces(currentHeight, fooCol, avoidPts):
	# Get only valid repulsion points
	fooAvoidPts = avoidPts[:, np.where((avoidPts[2] < currentHeight) & (avoidPts[2] > currentHeight-Z_DIFF_MAX))[0]]

	# Bail if no points to avoid
	if fooAvoidPts.shape[1] == 0:
		return(np.zeros((2), dtype=float))
	
	# Compare each avoid point
	zDiff = currentHeight - fooAvoidPts[2] # Difference in height
	posDiff = fooCol.currPos[:, None] - fooAvoidPts[:2] # Difference in XY position
	distance = magnitude(posDiff[:2]) # XY distance

	# Calculate force magnitude based on Z diff
	zDiffMag = np.ones_like(distance)
	calcZdiffSubset = np.where(zDiff > Z_DIFF_MIN)
	zDiffMag[calcZdiffSubset] = 1 - (zDiff[calcZdiffSubset] - Z_DIFF_MIN) / (Z_DIFF_MAX - Z_DIFF_MIN)
	zDiffMag[zDiff > Z_DIFF_MAX] = 0

	# Calculate force magnitude based on XY diff
	posDiffMag = np.ones_like(distance)
	calcDistSubset = np.where(zDiff > Z_DIFF_MIN)
	posDiffMag[calcDistSubset] = 1 - (distance[calcDistSubset] - POS_DIFF_MIN) / (POS_DIFF_MAX - POS_DIFF_MIN)
	posDiffMag[distance > POS_DIFF_MAX] = 0
	
	# Zero all forces where repel pt -> curr point slope < 45 deg
	zDiffMag[distance > zDiff*2] = 0.0

	repulsiveForces = PEAK_REPULSION_MAG * pow(posDiffMag, 2.0) * pow(zDiffMag, 2.0) * posDiff/distance
	return(np.sum(repulsiveForces, axis=1))

# Calculate attraction to other supports
def calculateAttractiveSupportForces(currentColumns, fooCol):
	attractiveForces = np.zeros_like(fooCol.sumAcc)

	for idx in range(len(currentColumns)):
		cmpCol = currentColumns[idx]
		if (cmpCol.currPos == fooCol.currPos).all(): continue # Do not compare point to itself

		posDiff = cmpCol.currPos - fooCol.currPos
		distance = magnitude(posDiff)

		if distance > SUPPORT_MAX_ATTRACTION_DIST:
			continue

		if distance < 1e-6: distance = 1e-6 # No super low distances (leads to massive acceleration)

		sizeAttractionMag = 1.0 / (1 +  np.abs(cmpCol.size - fooCol.size) * np.interp(
			cmpCol.size,
			[1, 2, 4, 10],
			[1.0, 1.0, 0.5, 0.05]
		))
		# sizeAttractionMag = 0.4 + 0.6 / (1 + np.sqrt(np.abs()))
		attraction = SUPPORT_ATTRACTION_CONSTANT * sizeAttractionMag / pow(distance, 2) 
		attractiveForces += attraction * posDiff/distance

	return attractiveForces

# Calculate pull to stay inside of the bounding box
def calculateBoundarySupportForces(fooCol):
	boundingBoxForce = np.zeros_like(fooCol.sumAcc)
	supportMargin = MARBLE_RAD*2
	for ax in range(len(boundingBoxForce)):
		if fooCol.currPos[ax] < -supportMargin + 0.0: boundingBoxForce[ax] -= fooCol.currPos[ax]
		if fooCol.currPos[ax] > supportMargin + BOUNDING_BOX[ax]: boundingBoxForce[ax] -= fooCol.currPos[ax]- BOUNDING_BOX[ax]
	boundingBoxForce *= SUPPORT_BOUNDARY_FORCE_MAG
	return(boundingBoxForce)

# Calculate pull towards center of box
def calculateCenteringForce(fooCol, targetRadius):
	centerForce = np.zeros_like(fooCol.sumAcc)
	for ax in range(len(centerForce)):
		centerForce[ax] = (BOUNDING_BOX[ax]/2 - fooCol.currPos[ax])

	centerForceMag = magnitude(centerForce)
	centerForce /= centerForceMag

	# Pull to target radius
	radDiff = centerForceMag - targetRadius
	if np.abs(radDiff) > PULL_TO_CENTER_MAXDIST:
		radDiff = np.sign(radDiff) * PULL_TO_CENTER_MAXDIST
	radDiff /= PULL_TO_CENTER_MAXDIST


	# Pull larger points to rad more strongly
	pullMag = np.interp(
		fooCol.size,
		[1, 2, 4, 20],
		[0.0, 0.1, 0.5, 1.0]
	)

	# Calculte final force to pull points towards target radius
	finalForce = centerForce * radDiff * PULL_TO_CENTER_MAG * pullMag

	return(finalForce)

# Calculate support paths from anchor and avoid points
def calculateSupports(anchorPts, avoidPts, visPath=None):
	# Tracking arrays
	supportNotPlaced = np.ones(anchorPts.shape[1], dtype=np.uint8) # If true, support has not been added yet
	completeColumns = []
	currentColumns = []

	# Init display output if requested
	if visPath != None:
		from PIL import Image, ImageDraw
		os.makedirs(visPath, exist_ok=True)
		imScale = 5
		plotImage = Image.new('RGB', (int(SIZE_X*imScale), int(SIZE_Y*imScale)))
		plotDraw = ImageDraw.Draw(plotImage)

	# Iterate over every layer height
	layerIdx = -1
	for currentHeight in np.arange(np.max(anchorPts[:, 2]), BASE_OF_MODEL, -SUPPORT_LAYER_HEIGHT):
		layerIdx += 1

		# Target radius of supports to pull towards
		targetRadius = np.interp(
			currentHeight,
			[10.0, 30.0],
			[OUTPUT_BASE_RAD, 6*MARBLE_RAD+SCREW_RAD],
		)

		# Iterate over existing columns to calculate motion
		for idx in range(len(currentColumns)):
			fooCol = currentColumns[idx]
			fooCol.prevPos = fooCol.currPos # Update prevpos
			fooCol.sumAcc[:] = 0 # Reset force

			# Get list of forces applied to each column
			attractiveForce = calculateAttractiveSupportForces(currentColumns, fooCol)
			repulsiveForce = calculateRepulsivesSupportForces(currentHeight, fooCol, avoidPts)
			boundaryForce = calculateBoundarySupportForces(fooCol)
			centerForce = calculateCenteringForce(fooCol, targetRadius)
			# centerForce = np.zeros_like(centerForce)

			# Calculate how important this motion is, prioritizing not hitting paths
			magnitudeOfPriority = magnitude(boundaryForce) + magnitude(repulsiveForce)
			if magnitudeOfPriority != 0.0:
				motionPriorityRatio =  magnitudeOfPriority / (magnitudeOfPriority + magnitude(attractiveForce) + magnitude(centerForce))
			else:
				motionPriorityRatio = 0.0
			# Calculate acceleration based purely on force and size
			fooCol.sumAcc = attractiveForce + boundaryForce + repulsiveForce + centerForce
			fooCol.sumAcc /= np.sqrt(np.clip(fooCol.size, 2, 10))
			accMag = magnitude(fooCol.sumAcc)
			if accMag > MAX_PARTICLE_ACC:
				fooCol.sumAcc = MAX_PARTICLE_ACC*fooCol.sumAcc/accMag
			# motionPriorityRatio + 

			# Calculate velocity
			fooCol.velocity = fooCol.velocity*PARTICLE_DRAG # Simulate drag to slow particles down
			fooCol.velocity += (fooCol.sumAcc)*(1.0-motionPriorityRatio) + (boundaryForce+repulsiveForce)*(motionPriorityRatio)
			
			# Limit vel
			velMag = pf.magnitude(fooCol.velocity)
			if velMag > MAX_PARTICLE_VEL:
				fooCol.velocity = MAX_PARTICLE_VEL*fooCol.velocity/velMag
			
			# Update position
			fooCol.currPos += fooCol.velocity


		# Merge supports which are within bounds
		newColumns = []
		for idx in range(len(currentColumns)):
			# Find matches
			fooCol = currentColumns[idx]
			for cmpIdx in range(len(currentColumns)):
				if idx == cmpIdx: 
					continue # Do not compare point to itself

				cmpCol = currentColumns[cmpIdx]
				posDiff = cmpCol.currPos - fooCol.currPos
				distance = magnitude(posDiff)

				if distance < MERGE_RAD:# Merging woo
					if cmpCol.mergingInto != -1: # Match has existing merge, join that 
						fooCol.mergingInto = cmpCol.mergingInto
					else: # Make new column
						fooCol.mergingInto = len(newColumns)
						newColumns.append(column(fooPt[:2], 0))
					continue

		# Calculate new merged columns
		for fooCol in currentColumns:
			if fooCol.mergingInto == -1: 
				# Only save current position if  not merging
				fooCol.posHist.append(np.array([fooCol.currPos[0], fooCol.currPos[1], currentHeight]))
				continue # Not merging


			mergeCol = newColumns[fooCol.mergingInto]
			mergeCol.currPos = (mergeCol.currPos*mergeCol.size + fooCol.currPos*fooCol.size) / (mergeCol.size + fooCol.size) # Take weighted average of positions
			mergeCol.velocity = (mergeCol.velocity*mergeCol.size + fooCol.velocity*fooCol.size) / (mergeCol.size + fooCol.size) # Take size weighted average of velocities
			mergeCol.size += fooCol.size # Sum sizes
			mergeCol.prevPos = mergeCol.currPos # Update previous position
			mergeCol.posHist = [np.array([mergeCol.currPos[0], mergeCol.currPos[1], currentHeight])] # Set position history

			# print(f"{fooCol.mergingInto} {fooCol.size} | {fooCol.velocity} -> {mergeCol.velocity}") # Debug print statement

		# Eliminate merged columns
		delIdx = 0
		while delIdx < len(currentColumns):
			fooCol = currentColumns[delIdx]
			if fooCol.mergingInto == -1: # Not merging, continuing
				delIdx += 1
				continue
			# Set final position

			mergeCol = newColumns[fooCol.mergingInto]
			fooCol.mergedSize = mergeCol.size # Save final size for later generation
			fooCol.posHist.append([mergeCol.currPos[0], mergeCol.currPos[1], currentHeight]) # Set position history

			mergeCol.mergedFrom.append(len(completeColumns)) # Record index in completeColumns of parent column
			completeColumns.append(currentColumns.pop(delIdx)) # Move current column to history

		
		# Remove empty columns
		fooIdx = 0
		while fooIdx < len(newColumns)-1:
			if newColumns[fooIdx].size == 0:
				newColumns.pop(fooIdx)
			else:
				fooIdx += 1
		
		# Append new columns to existing set
		currentColumns += newColumns


		# Iterate through new support queue, add if below current height
		for idx in np.where((supportNotPlaced) & (anchorPts[2] > currentHeight))[0]:
			fooPt = anchorPts[:, idx]
			currentColumns.append(column(fooPt[:2], 1)) # Add new column
			currentColumns[-1].posHist.append(fooPt)
			supportNotPlaced[idx] = 0 # Do not place point again



		print("{:4d} {:4.10f} {:4d} {:4d} {:4d}".format(
			layerIdx,
			currentHeight, 
			len(completeColumns),
			len(currentColumns),
			sum(supportNotPlaced),
			))
			
		if visPath != None:
			plotDraw.rectangle((0, 0, SIZE_X*imScale, SIZE_Y*imScale), fill=(0,0,0))

			circleRad = 0.5
			for fooPt in currentColumns:
				circleRad = getColumnRad(fooPt.size)
				# drawline = (fooPt.currPos[0]*imScale-circleRad, fooPt.currPos[1]*imScale-circleRad, fooPt.prevPos[0]*imScale+circleRad, fooPt.prevPos[1]*imScale+circleRad)
				# plotDraw.line(drawline, fill=currentColor, width=5)
		
				drawEllipse = (fooPt.currPos[0]*imScale-circleRad, fooPt.currPos[1]*imScale-circleRad, fooPt.currPos[0]*imScale+circleRad, fooPt.currPos[1]*imScale+circleRad)
				velMag = int(512*magnitude(fooPt.velocity)/MAX_PARTICLE_VEL)
				plotDraw.ellipse(drawEllipse, fill=(0, np.clip(velMag, 0, 255), np.clip(255-velMag, 0, 255)), width=5)
			
			for fooPt in np.swapaxes(avoidPts, 0, 1):
				zDiff = currentHeight - fooPt[2]
				if zDiff > Z_DIFF_MAX: continue
				if zDiff < 0: continue

				colMag = 1.0 - np.clip((zDiff - Z_DIFF_MIN) / (Z_DIFF_MAX - Z_DIFF_MIN), 0.0, 1.0)
				circleRad = 1
				drawEllipse = (fooPt[0]*imScale-circleRad, fooPt[1]*imScale-circleRad, fooPt[0]*imScale+circleRad, fooPt[1]*imScale+circleRad)
				plotDraw.ellipse(drawEllipse, fill=(50+int(205*colMag), 0, 0), width=5)
				
			plotImage.save(f"{visPath}/layer_{layerIdx}.png")

	# Save all current columns to be export
	completeColumns += currentColumns

	if np.sum(supportNotPlaced) > 0:
		print(f"Failed to place {np.sum(supportNotPlaced)} supports, check your Z height variables")
		exit()

	return completeColumns

# Calculate radius of column from size
def getColumnRad(size):
	# fooRad = TRACK_SUPPORT_RAD*np.sqrt(size)

	# fooRad = TRACK_SUPPORT_RAD*np.log(size*np.e)
	# if fooRad > TRACK_SUPPORT_MAX_RAD: fooRad = TRACK_SUPPORT_MAX_RAD

	sizeFrac = np.interp(
		size,
		[1, 6, 15],
		[0.0, 0.5, 1.0]
	)
	
	fooRad = sizeFrac*(TRACK_SUPPORT_MAX_RAD - TRACK_SUPPORT_RAD) + TRACK_SUPPORT_RAD
	return(fooRad)

# Generate supports from calculated columns
def generateSupports(supportCols):
	supports = sphere(0)
	basePositions = [] # List of points and rads to generate the base plat from

	for fooCol in supportCols:
		# Calculate size of each disk
		size = fooCol.size
		mergedSize = fooCol.mergedSize
		if mergedSize == -1: mergedSize = fooCol.size * 2 # If a column made it to the base, sent the end radius to 2x the initial
		ptCnt = len(fooCol.posHist)
		
		# Calculate size of each profile
		sizeList = np.linspace(getColumnRad(size), getColumnRad(mergedSize), ptCnt)
		if ptCnt > MERGE_SMOOTH_PTS:
			sizeList[:] = getColumnRad(fooCol.size)
			sizeList[-MERGE_SMOOTH_PTS:] = np.linspace(getColumnRad(size), getColumnRad(mergedSize), MERGE_SMOOTH_PTS)
		
		# Generate the profiles
		outProfiles = []
		
		for fooIdx in range(ptCnt):
			fooPos = fooCol.posHist[fooIdx]

			# outProfiles.append(sphere(getColumnRad(sizeList[fooIdx]), _fn=UNIVERSAL_FN).translate(fooCol.posHist[0]))
			outProfiles.append(sphere(sizeList[fooIdx], _fn=UNIVERSAL_FN).translate(fooPos))

			# if fooIdx == 0 and len(fooCol.mergedFrom) == 0:
			# 	outProfiles.append(sphere(TRACK_RAD, _fn=UNIVERSAL_FN).translate(fooPos))
			# else:
			# 	fooProfile = linear_extrude(0.05)(circle(getColumnRad(sizeList[fooIdx]))).translate(fooPos)
			# 	outProfiles.append(fooProfile)

		# Chain hull profiles together
		supports += chain_hull()(*outProfiles)
		# for foo in outProfiles: supports += foo # DEBUG

		if fooCol.mergingInto == -1:
			basePositions.append(fooCol.posHist[-1])

		# # Join columns
		# for mergeIdx in fooCol.mergedFrom:
		# 	mergedCol = supportCols[mergeIdx]
		# 	if ptCnt == 0:continue			
		# 	outProfiles = [
		# 		linear_extrude(0.05)(circle(getColumnRad(mergedCol.mergedSize))).translate(fooCol.posHist[0]),
		# 		linear_extrude(0.05)(circle(getColumnRad(mergedCol.mergedSize))).translate(mergedCol.posHist[-1]),
		# 	]
		# 	supports += chain_hull()(*outProfiles)

	baseSpheres = []
	for foo in basePositions:
		foo[2] = BASE_OF_MODEL - BASE_THICKNESS/2
		baseSpheres.append(sphere(BASE_THICKNESS/2, _fn=UNIVERSAL_FN).translate(foo))
	supports += conv_hull()(*baseSpheres)

	# Cutout for motor
	if MOTOR_TYPE == 'SMALL_DC':
		cutout = cylinder(8, 1.5, 1.5, _fn=HIGHER_RES_FN)
		frontX = 12
		frontY = 10
		maxHeight = 30
		lipDepth = 2
		lipThickness = 2

		faceCutout = cube([frontX, frontY, maxHeight])
		# faceCutout += cube([frontX-lipDepth*2, frontY-lipDepth*2, maxHeight+lipThickness*2]).translate([lipDepth, lipDepth, 0])

		cutout += faceCutout.translate([-frontX/2, -frontY/2, -maxHeight-lipThickness+BASE_OF_MODEL])
		cutout += cylinder(maxHeight+lipThickness*2, frontY/2-lipDepth, frontY/2-lipDepth).translateZ(-maxHeight-lipThickness+BASE_OF_MODEL)

	elif MOTOR_TYPE == 'NEMA17':
		lipThickness = 2
		maxHeight = 30
		# cutout = cylinder(maxHeight+lipThickness*2, 4.0, 4.0, _fn=HIGHER_RES_FN).translateZ(-maxHeight-lipThickness+BASE_OF_MODEL)
		cutout = cylinder(maxHeight+lipThickness*2, 11.05, 11.05, _fn=HIGHER_RES_FN).translateZ(-maxHeight-lipThickness+BASE_OF_MODEL)
		cutout += cylinder(1.0, 12.0, 11.0, _fn=HIGHER_RES_FN).translateZ(-maxHeight-lipThickness+BASE_OF_MODEL)

		# cutout += cylinder(maxHeight, 11.1, 11.1, _fn=HIGHER_RES_FN).translateZ(-maxHeight-lipThickness+BASE_OF_MODEL)

	cutout += generateCutoutForPrinting().translateZ(BASE_OF_MODEL - BASE_THICKNESS - 1e-3) # Cut out vent holes for SLA printing


	supports -= cutout.translate(SCREW_POS)

	return(supports)
