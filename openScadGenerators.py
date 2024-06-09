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
import os

from defs import *
from shared import *
import positionFuncs as pf

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

# Generate the actual rotating part of the screw lift
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

# Get the support anchors for the track
def calculateSupportAnchorsForPath(path, rotations):
	lowerDist = TRACK_SUPPORT_RAD*2
	trackToPathDist = MARBLE_RAD + TRACK_RAD

	# Calculate the offset of each of the supporting points relative to the frame
	rightSupportOffset = [0, trackToPathDist*np.cos(TRACK_CONTACT_ANGLE), -trackToPathDist*np.sin(TRACK_CONTACT_ANGLE)-lowerDist]
	leftSupportOffset = [0, -trackToPathDist*np.cos(TRACK_CONTACT_ANGLE), -trackToPathDist*np.sin(TRACK_CONTACT_ANGLE)-lowerDist]

	supportSectionsCount = int(np.ceil(path.shape[1]/2))
	anchorPts = np.zeros((3, supportSectionsCount*2), dtype=np.double)
	for ii in range(0, supportSectionsCount*2, 2):
		anchorPts[:, ii] = path[:, ii] + pf.doRotationMatrixes(rightSupportOffset, [rotations[1, ii], 0, rotations[0, ii]])
		anchorPts[:, ii+1] = path[:, ii] + pf.doRotationMatrixes(leftSupportOffset, [rotations[1, ii], 0, rotations[0, ii]])

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
	posDiff = fooAvoidPts[:2] - fooCol.currPos[:, None] # Difference in XY position
	distance = magnitude(posDiff[:2]) # XY distance

	# Calculate force magnitude based on Z diff
	zDiffMag = np.ones_like(distance)
	zDiffMag[zDiff > Z_DIFF_MIN] = 1 - (zDiff - Z_DIFF_MIN) / (Z_DIFF_MAX - Z_DIFF_MIN)
	zDiffMag[zDiffMag > Z_DIFF_MAX] = 0

	# Calculate force magnitude based on XY diff
	posDiffMag = np.ones_like(distance)
	posDiffMag[distance > POS_DIFF_MIN] = 1 - (distance - POS_DIFF_MIN) / (POS_DIFF_MAX - POS_DIFF_MIN)
	posDiffMag[posDiffMag > POS_DIFF_MAX] = 0
	
	repulsiveForces = PEAK_REPULSION_MAG * pow(posDiffMag, 3) * pow(zDiffMag, 3) * posDiff/distance
	return(np.sum(repulsiveForces, axis=1))

# Calculate attraction to other supports
def calculateAttractiveSupportForces(currentColumns, fooCol):
	attractiveForces = np.zeros_like(fooCol.sumAcc)

	for idx in range(len(currentColumns)):
		cmpCol = currentColumns[idx]
		if (cmpCol.currPos == fooCol.currPos).all(): continue # Do not compare point to itself

		posDiff = cmpCol.currPos - fooCol.currPos
		distance = magnitude(posDiff)

		if distance < 1e-6: distance = 1e-6 # No super low distances (leads to massive acceleration)

		# if distance > SUPPORT_BOUNDARY_FORCE_MAG: # Ignore out of range
		# 	continue

		attraction = SUPPORT_ATTRACTION_CONSTANT * cmpCol.size / pow(distance, 2) 
		attractiveForces += attraction * posDiff/distance

	return attractiveForces

# Calculate pull to stay inside of the bounding box
def calculateBoundarySupportForces(fooCol):
	boundingBoxForce = np.zeros_like(fooCol.sumAcc)
	for ax in range(len(boundingBoxForce)):
		if fooCol.currPos[ax] < 0.0: boundingBoxForce[ax] -= fooCol.currPos[ax]
		if fooCol.currPos[ax] > BOUNDING_BOX[ax]: boundingBoxForce[ax] -= fooCol.currPos[ax]- BOUNDING_BOX[ax]
	boundingBoxForce *= SUPPORT_BOUNDARY_FORCE_MAG
	return(boundingBoxForce)

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
		plotImage = Image.new('RGB', (SIZE_X*imScale, SIZE_Y*imScale))
		plotDraw = ImageDraw.Draw(plotImage)

	# Iterate over every layer height
	layerIdx = -1
	for currentHeight in np.arange(np.max(anchorPts[:, 2]), BASE_OF_MODEL, -SUPPORT_LAYER_HEIGHT):
		layerIdx += 1

		# Iterate through new support queue, add if below current height
		for idx in np.where((supportNotPlaced) & (anchorPts[2] > currentHeight))[0]:
			fooPt = anchorPts[:, idx]
			currentColumns.append(column(fooPt[:2], 1)) # Add new column
			currentColumns[-1].posHist.append(fooPt)
			supportNotPlaced[idx] = 0 # Do not place point again

		# Iterate over existing columns to calculate motion
		for idx in range(len(currentColumns)):
			fooCol = currentColumns[idx]
			fooCol.prevPos = fooCol.currPos # Update prevpos
			fooCol.sumAcc[:] = 0 # Reset force

			# Get list of forces applied to each column
			attractiveForce = calculateAttractiveSupportForces(currentColumns, fooCol)
			repulsiveForce = calculateRepulsivesSupportForces(currentHeight, fooCol, avoidPts)
			boundaryForce = calculateBoundarySupportForces(fooCol)

			# Calculate acceleration
			fooCol.sumAcc = attractiveForce + boundaryForce + repulsiveForce

			# Limit acceleration
			accMag = magnitude(fooCol.sumAcc)
			if accMag > MAX_MOVE_PER_LAYER:
				fooCol.sumAcc = MAX_MOVE_PER_LAYER*fooCol.sumAcc/accMag
			
			# Calculate velocity
			fooCol.velocity += fooCol.sumAcc/fooCol.size
				
			# Limit vel
			velMag = pf.magnitude(fooCol.velocity)
			if velMag > MAX_MOVE_PER_LAYER:
				fooCol.velocity = MAX_MOVE_PER_LAYER*fooCol.velocity/velMag
			
			# Update position
			fooCol.currPos += fooCol.velocity
			fooCol.posHist.append(np.array([fooCol.currPos[0], fooCol.currPos[1], currentHeight]))


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

		# Calculate new merged columns
		for fooCol in currentColumns:
			if fooCol.mergingInto == -1: continue # Not merging

			mergeCol = newColumns[fooCol.mergingInto]
			mergeCol.currPos = (mergeCol.currPos*mergeCol.size + fooCol.currPos*fooCol.size) / (mergeCol.size + fooCol.size) # Take average of positions
			mergeCol.velocity = (mergeCol.velocity*mergeCol.size + fooCol.velocity*fooCol.size) / (mergeCol.size + fooCol.size) # Take size weighted average of velocities
			mergeCol.prevPos = mergeCol.currPos # Update previous position
			mergeCol.size += fooCol.size # Sum sizes

		# Eliminate merged columns
		delIdx = 0
		while delIdx < len(currentColumns):
			fooCol = currentColumns[delIdx]
			if fooCol.mergingInto == -1: # Not merging, continuing
				delIdx += 1
				continue

			mergeCol = newColumns[fooCol.mergingInto]
			fooCol.mergedSize = mergeCol.size # Save final size for later generation
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

		print("{:4.10f} {:4d} {:4d} {:4d}".format(
			currentHeight, 
			len(completeColumns),
			len(currentColumns),
			999
			))
			
		if visPath != None:
			plotDraw.rectangle((0, 0, SIZE_X*imScale, SIZE_Y*imScale), fill=(0,0,0))

			circleRad = 0.5
			for fooPt in currentColumns:
				circleRad = fooPt.size/2
				# drawline = (fooPt.currPos[0]*imScale-circleRad, fooPt.currPos[1]*imScale-circleRad, fooPt.prevPos[0]*imScale+circleRad, fooPt.prevPos[1]*imScale+circleRad)
				# plotDraw.line(drawline, fill=currentColor, width=5)
		
				drawEllipse = (fooPt.currPos[0]*imScale-circleRad, fooPt.currPos[1]*imScale-circleRad, fooPt.currPos[0]*imScale+circleRad, fooPt.currPos[1]*imScale+circleRad)
				plotDraw.ellipse(drawEllipse, fill=(0, 0, 255), width=5)
			
			for fooPt in np.swapaxes(avoidPts, 0, 1):
				zDiff = currentHeight - fooPt[2]
				if zDiff > Z_DIFF_MAX: continue
				if zDiff < 0: continue

				circleRad = 1
				drawEllipse = (fooPt[0]*imScale-circleRad, fooPt[1]*imScale-circleRad, fooPt[0]*imScale+circleRad, fooPt[1]*imScale+circleRad)
				plotDraw.ellipse(drawEllipse, fill=(255, 0, 0), width=5)
				
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
	fooRad = TRACK_SUPPORT_RAD*np.log(size*np.e)
	if fooRad > TRACK_SUPPORT_MAX_RAD: fooRad = TRACK_SUPPORT_MAX_RAD
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
		sizeList = np.linspace(size, mergedSize, ptCnt)
		if ptCnt > MERGE_SMOOTH_PTS:
			sizeList[:] = fooCol.size
			sizeList[-MERGE_SMOOTH_PTS:] = np.linspace(size, mergedSize, MERGE_SMOOTH_PTS)
		
		# Generate the profiles
		outProfiles = []
		for fooIdx in range(ptCnt):
			fooPos = fooCol.posHist[fooIdx]
			fooProfile = linear_extrude(0.05)(circle(getColumnRad(sizeList[fooIdx]))).translate(fooPos)
			outProfiles.append(fooProfile)

		# Chain hull profiles together
		supports += chain_hull()(*outProfiles)
		# for foo in outProfiles: supports += foo # DEBUG


		# Join columns
		for mergeIdx in fooCol.mergedFrom:
			mergedCol = supportCols[mergeIdx]

			if ptCnt == 0:continue
			
			outProfiles = [
				linear_extrude(0.05)(circle(getColumnRad(mergedCol.mergedSize))).translate(fooCol.posHist[0]),
				linear_extrude(0.05)(circle(getColumnRad(mergedCol.mergedSize))).translate(mergedCol.posHist[-1]),
			]
			supports += chain_hull()(*outProfiles)
	return(supports)