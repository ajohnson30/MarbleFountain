import numpy as np
from scipy.interpolate import splprep, splev
from scipy.spatial.distance import cdist
from scipy.linalg import norm
from random import random
from copy import deepcopy
import pickle as pkl
import os

from defs import *

def getAng(pt1, pt2): return(np.arctan2(pt2[1] - pt1[1], pt2[0] - pt1[0]))

def angDiff(a1, a2):
	a = a2 - a1
	if a > np.pi: a -= 2*np.pi
	if a < -np.pi: a += 2*np.pi
	return a

# Generate random initial path
def randomPath(ptCnt, box, pathIdx):
	path = np.zeros((3, ptCnt), dtype=np.double)

	angle = getPathAnchorAngle(pathIdx)
	
	if LESS_RANDOM_INIT_PATH:
		# Sort of random points
		for idx in range(3):
			randPts = np.random.random(RANDOM_CNT+2)
			randPts *= box[idx]
			if idx == 0:
				startPos = np.cos(angle)*(SCREW_RAD + PT_SPACING) + box[idx]/2
				randPts[0] = startPos
				randPts[-1] = startPos
			elif idx == 1:
				startPos = np.sin(angle)*(SCREW_RAD + PT_SPACING) + box[idx]/2
				randPts[0] = startPos
				randPts[-1] = startPos
			path[idx] = np.interp(np.linspace(0, RANDOM_CNT+1, ptCnt), np.arange(RANDOM_CNT+2), randPts)

			path[idx] += 0.5 - np.random.random(len(path[idx]))
	else:
		# Fully random points
		path[:3] = np.random.random(path[:3].shape)
		path[0] *= box[0]
		path[1] *= box[1]
		path[2] *= box[2]
	
	return(path)

def getPathAnchorAngle(pathIdx):
	angle = np.pi*2*pathIdx/PATH_COUNT
	if PATH_COUNT%2 == 0: 
		angle += np.pi/(PATH_COUNT)
	if PATH_COUNT==5:
		angle += np.pi/4
	return angle

# Pull towards bounding box
def pushTowardsBoundingBox(pts, box, forceCurve, axCount = 2):
	outForces = np.zeros_like(pts)
	zeros = np.zeros_like(pts[0])
	
	boxSet = np.tile(box, (pts.shape[1], 1)).T
	negForces = np.interp(pts-boxSet, *forceCurve)
	posForces = np.interp(-pts, *forceCurve)
	outForces = np.where(posForces > negForces, posForces, -negForces)

	# Old method (pre interpolation)
	# outForces[np.where(pts > boxSet)] -= (pts - boxSet)[np.where(pts > boxSet)]
	# outForces[np.where(pts < 0.0)] -= pts[np.where(pts < 0.0)]
	# for ax in range(axCount):
	# 	outForces[ax] -= np.min([pts[ax], zeros])
	# 	outForces[ax] += np.min([box[ax] - pts[ax], zeros])
	# outForces *= forcePerDist
	# outForces = np.clip(outForces, -maxForcePerAxis, maxForcePerAxis)

	return(outForces)

# Pull towards Z position
def pullTowardsTargetHeights(pts, zTargetPositions, forcePerDist, maxForce=5):
	outForces = np.zeros_like(pts)
	outForces[2] = forcePerDist * (zTargetPositions - pts[2])
	outForces[2] = np.clip(outForces[2], -maxForce, maxForce)
	return outForces

# Pull towards Z position
def pullTowardsTargetSlope(pts, targetPointDrop, forcePerDist, maxForce=5):
	outForces = np.zeros_like(pts)
	ptDiff = pts[2, 1:] - pts[2, :-1]
	# print(ptDiff - 2*targetPointDrop)
	outForces[2, :-1] = forcePerDist*(ptDiff - targetPointDrop)
	outForces[2] = np.clip(outForces[2], -maxForce, maxForce)
	return outForces

# Shorthand for magnitude of vector
def magnitude(vect):
	if len(vect.shape) == 2:
		magnitude = np.sqrt(np.sum(pow(vect,2), axis=0))
		magnitude[magnitude == 0.0] = 1e-20
		return(magnitude)
	else:
		return(np.sqrt(np.sum(pow(vect,2))))

# Get distance betweeen each pair of points
def getPathDists(path):
	return magnitude(path[:, 1:] - path[:, :-1])

# Normalize distances between points
def normalizePathDists(path, targDist, forcePerDist, maxForce = 5.0, pointOffset = 1, dropZ = True):
	if dropZ:
		axisCap = 2
	else:
		axisCap = 3
	pathDiffs = path[:axisCap, pointOffset:] - path[:axisCap, :-pointOffset]
	pathDists = magnitude(pathDiffs)

	pathNorms = pathDiffs / pathDists

	forceMags = (targDist - pathDists) * forcePerDist / 2

	# forceMags = np.max([forceMags, np.zeros_like(forceMags)-10], axis=0)

	outForces = np.zeros_like(path)
	outForces[:axisCap, :-pointOffset] -= forceMags * pathNorms
	outForces[:axisCap, pointOffset:] += forceMags * pathNorms

	# # Make force magnitudes constant
	# netForceMags = np.zeros(path.shape[1])
	# netForceMags[:-pointOffset] += np.abs(forceMags)
	# netForceMags[pointOffset:] += np.abs(forceMags)
	# netForceMags = np.clip(netForceMags, -maxForce, maxForce)
	
	# outForces = netForceMags * outForces/np.linalg.norm(outForces)

	# print(forceMags)
	return outForces

# Repel away from paths
def repelPoints(path, repelPts, peakForce, cutOffDist):
	outForces = np.zeros_like(path)
	for ptIdx in range(path.shape[1]):
		fooPt = path[:, ptIdx]

		ptDiffs = fooPt[:, None] - repelPts
		ptDists = magnitude(ptDiffs)
		ptForceMags = (peakForce/cutOffDist) * np.max([cutOffDist - ptDists, np.zeros_like(ptDists)], axis=0)
		ptDists[ptDists == 0.0] = 1e-6 # Handle overlapping points
		outForces[:, ptIdx] = np.sum(ptForceMags * (ptDiffs / ptDists), axis=1) 

	return outForces

# Repel away from own path
def repelPathFromSelf(path, dropAdjacentPointCnt, peakForce, cutOffDist):
	outForces = np.zeros_like(path)
	for ptIdx in range(path.shape[1]):
		fooPt = path[:, ptIdx]
		pathSubset = path[:, ptIdx+dropAdjacentPointCnt+1:]
		if ptIdx > dropAdjacentPointCnt: 
			pathSubset = np.concatenate([
				path[:, :ptIdx-dropAdjacentPointCnt],
				pathSubset,
			], axis=1)

		ptDiffs = fooPt[:, None] - pathSubset
		ptDists = magnitude(ptDiffs)
		ptForceMags = (peakForce/cutOffDist) * np.clip(cutOffDist - ptDists, 0.0, cutOffDist)

		outForces[:, ptIdx] = np.sum(ptForceMags * (ptDiffs / ptDists), axis=1) 

	return outForces

# Limit path angle
def correctPathAngle(path, minAng, maxAng, forcePerRad, maxForce=5, diffPointOffsetCnt=1, flatten=True):
	# NOTE: 3D mode seems to be bugged, do not enable atm
	if flatten:
		path = deepcopy(path)
		path[2] = 0

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

	# Calculate magnitude of forces
	forceMags = np.zeros_like(angles)
	forceMags = np.max([minAng-angles, forceMags], axis=0)
	forceMags = np.min([maxAng-angles, forceMags], axis=0)
	forceMags *= forcePerRad

	outForces = np.zeros_like(path)

	# Apply force on center of each angle
	forceNormalVect = (prevNorm + nextNorm) / 2
	forceNormalVect = forceNormalVect/magnitude(forceNormalVect)
	outForces[:, diffPointOffsetCnt:-diffPointOffsetCnt] += forceMags*forceNormalVect/2
	
	# # Apply inline force on each adjacent particle
	# outForces[:, :-(diffPointOffsetCnt*2)] += forceMags*prevNorm
	# outForces[:, (diffPointOffsetCnt*2):]  += forceMags*nextNorm

	# # Push/pull adjacent particles towards or away from each other
	# # This helps propagate desired change along path
	# # Could definitely be improved by forcibly rotating each point to the precise correct position, but I don't want to do that math rn
	# prevToNextVect = path[:, 2:] - path[:, :-2]
	# prevToNextNorm = prevToNextVect/magnitude(prevToNextVect)
	# outForces[:, :-2] -= forceMags*prevToNextNorm
	# outForces[:, 2:]  += forceMags*prevToNextNorm
	
	# Use cross product to calculate appropriate vectors to pull apart
	crossProdNorm = np.cross(nextNorm, prevNorm, axis=0)
	crossProdNorm = crossProdNorm/magnitude(crossProdNorm)
	
	testVects = np.zeros_like(path)
	outForces[:, (diffPointOffsetCnt*2):]  += forceMags * np.cross(nextNorm, crossProdNorm, axis=0)
	outForces[:, :-(diffPointOffsetCnt*2)] -= forceMags * np.cross(prevNorm, crossProdNorm, axis=0)

	outForceMags = magnitude(outForces)
	outForceVels = outForces/outForceMags
	outForceMags = np.clip(outForceMags, -maxForce, maxForce)

	return outForceVels*outForceMags





# Calculate the tangent circle made using each set of three points
def approximatePathCurvature(path, offset=1):
	# (This portion written by Claude AI, seems very close to working but I mostly only need XY)
	N = path.shape[1]
	
	if N <= 2 * offset:
		return np.zeros_like(path[:, :1])

	# Create views of the path array for p1, p2, and p3
	p1 = path[:, :-2*offset]
	p2 = path[:, offset:-offset]
	p3 = path[:, 2*offset:]

	# Calculate vectors
	v1 = p2 - p1
	v2 = p3 - p1
	v3 = p3 - p2

	# Calculate cross products
	cross = np.cross(v1, v2, axis=0)

	# Calculate radii
	numerator = np.linalg.norm(v1, axis=0) * np.linalg.norm(v2, axis=0) * np.linalg.norm(v3, axis=0)
	denominator = 2 * np.abs(cross)
	radii = np.where(denominator != 0, numerator / denominator, np.inf)

	# Calculate vectors to circle centers 
	normCenterVect = (np.cross(v3, cross, axis=0) * np.linalg.norm(v1, axis=0)**2 + 
				np.cross(cross, v1, axis=0) * np.linalg.norm(v3, axis=0)**2) / (4 * cross**2)
	
	# Handle cases where cross is zero (points are collinear)
	normCenterVect = np.where(cross == 0, np.inf, normCenterVect)
	normCenterVect /= np.linalg.norm(normCenterVect, axis=0)
	
	# Calculate vectors to circle centers
	centerVect = radii*normCenterVect
	
	centerVectPrev = centerVect+v1
	centerVectPrev /= np.linalg.norm(centerVectPrev, axis=0)

	centerVectNext = centerVect-v3
	centerVectNext /= np.linalg.norm(centerVectNext, axis=0)

	# Return radius and vectors to center from p1, p2, & p3
	return radii, centerVectPrev, normCenterVect, centerVectNext

# Calculate the tangent circle made using each set of three points in XY plane
def approximatePathCurvatureXY(path, offset=1, includeCurvatureDir=False):
	N = path.shape[1]
	
	if N <= 2 * offset:
		return np.zeros_like(path[:, :1])

	# Create views of the path array for p1, p2, and p3
	p1 = path[:2, :-2*offset]
	p2 = path[:2, offset:-offset]
	p3 = path[:2, 2*offset:]

	# Calculate vectors
	v1 = p2 - p1
	v2 = p3 - p1
	v3 = p3 - p2

	# Calculate cross products
	cross = np.cross(v1, v2, axis=0)

	# Calculate radii
	numerator = np.linalg.norm(v1, axis=0) * np.linalg.norm(v2, axis=0) * np.linalg.norm(v3, axis=0)
	denominator = 2 * np.abs(cross)
	denominator = np.where(denominator !=0, denominator, 1.0)
	radii = np.where(denominator != 0, numerator / denominator, np.inf) # Handle zero case

	# Calculate vector to circle center
	normCenterVect = -v1/np.linalg.norm(v1, axis=0) + v3/np.linalg.norm(v3, axis=0)
	normCenterVectNormal = np.linalg.norm(normCenterVect, axis=0)
	normCenterVect /= np.where(normCenterVectNormal != 0.0, normCenterVectNormal, 1e-9)

	# Calculate vectors to circle centers
	centerVect = radii*normCenterVect
	
	centerVectPrev = centerVect+v1
	centerVectPrev /= np.linalg.norm(centerVectPrev, axis=0)

	centerVectNext = centerVect-v3
	centerVectNext /= np.linalg.norm(centerVectNext, axis=0)
	
	if includeCurvatureDir:
		return radii, centerVectPrev, normCenterVect, centerVectNext, np.sign(cross)

	# Return radius and vectors to center from p1, p2, & p3
	return radii, centerVectPrev, normCenterVect, centerVectNext

# Helper function to calculate how long it's been since the path changed directions
# Written entirely by Claude AI
def distance_to_sign_change(arr):
    # Ensure the input is a NumPy array
    arr = np.asarray(arr)
    
    # # Check if the array contains only +1 and -1
    # if not np.all(np.abs(arr) == 1):
    #     raise ValueError("Array must contain only +1 and -1")

    # Calculate the difference between adjacent elements
    diff = np.diff(arr)
    
    # Find indices where the sign changes (diff will be +2 or -2)
    change_indices = np.where(np.abs(diff) == 2)[0]
    
    if len(change_indices) == 0:
        # If no sign changes, return an array of the maximum possible distance
        return np.full_like(arr, len(arr) - 1)
    
    # Calculate distances to the left and right sign changes
    dist_left = np.arange(len(arr))[:, None] - change_indices
    dist_right = change_indices - np.arange(len(arr))[:, None]
    
    # Combine distances, ignoring negative values
    dist = np.where(dist_left < 0, dist_right, dist_left)
    dist = np.where(dist_right < 0, dist_left, dist)
    
    # Return the minimum distance for each index
    return np.min(dist, axis=1)

# Correct path curvature by calculating radius of tangent cirle
# This is more resiliant to changes in point spacing
def update_path_curvature(path, min_radius, max_radius, updateMag=1.0, maxMag=5.0, offset=1):
	N = path.shape[1]

	if N <= 2 * offset:
		return updates

	# Get normal vectors to center points of curvature
	radii, centerVectPrev, normCenterVect, centerVectNext = approximatePathCurvatureXY(path, offset=offset)

	if type(min_radius) == np.ndarray:
		min_radius = min_radius[offset:-offset]
	if type(max_radius) == np.ndarray:
		max_radius = max_radius[offset:-offset]

	# if curvInflectionLimits != None:
	# 	min_radius = np.interp(
	# 		distance_to_sign_change(curvatureSign),
	# 		[curvInflectionLimits[0], curvInflectionLimits[1]],
	# 		[curvInflectionLimits[2] + min_radius, min_radius]
	# 	)
	# Calculate magnitude of force correction
	forceMags = np.zeros_like(radii)
	forceMags = np.max([min_radius-radii, forceMags], axis=0)
	forceMags = np.min([max_radius-radii, forceMags], axis=0)
	forceMags *= updateMag
	forceMags = np.clip(forceMags, -maxMag, maxMag) # Clip to max update


	updates = np.zeros_like(path)
	updates[:2, :-2*offset] -= forceMags * centerVectPrev / 2
	updates[:2, offset:-offset] += forceMags * normCenterVect / 2
	updates[:2, 2*offset:] -= forceMags * centerVectNext / 2
	
	return updates

# Calculate slope of path (rise / run)
def calcPathSlope(path):
	zDiffs = np.diff(path[2])
	xyDist = magnitude(np.diff(path[:2], axis=1))
	return zDiffs/xyDist

# Smooth out change in slope
def correctSlopeChange(path, forceMag = 1.0, slopeErrMag=0.2, upwardsForceMag = None, offset=1):
	outForces = np.zeros_like(path)

	zDiffs = path[2, offset*2:] - path[2, :-offset*2]
	xyDist = magnitude(path[:2, offset*2:] - path[:2, :-offset*2])
	slope = zDiffs/xyDist

	averageSlope = np.average(slope)

	# plt.plot(averageSlope*np.ones_like(slope))
	# plt.plot(slope)
	# plt.plot(np.diff(slope))
	# plt.show()

	if forceMag > 0.0 or upwardsForceMag:
		if upwardsForceMag == None:
			upwardsForceMag = forceMag
		
		slopeErr = (slope - averageSlope)/averageSlope

		# slopeErr = np.where(slopeErr < 0, slopeErr*2, slopeErr) # Double magnitude of overly flat slope

		slopeForce = np.where(slopeErr > 0, slopeErr*upwardsForceMag, slopeErr*forceMag)

		outForces[2, offset*2:] += slopeForce/2
		outForces[2, :-offset*2] -= slopeForce/2


	if slopeErrMag > 0.0:
		maxSlopeDelta = averageSlope
		maxSlopeDeltaCap = maxSlopeDelta*2
		slopeDiff = slope[2:] - slope[:-2]
		
		slopeDiffErrMag = np.interp(
			slopeDiff,
			[-maxSlopeDeltaCap, -maxSlopeDelta, maxSlopeDelta, maxSlopeDeltaCap],
			[-slopeErrMag, 0.0, 0.0, slopeErrMag]
		)
		outForces[2, offset+1:-offset-1] = slopeDiffErrMag
		# outForces[2, offset:-offset] = slopeDiffErrMag

	return(outForces)

def preventUphillMotion(path, forceMag = 0.1):
	slopeDownForce = np.zeros_like(path)
	zVal = path[2]
	zDiff = np.diff(zVal)
	zDiffAvg = np.average(zDiff)
	startIdx = None

	zTargetMax = deepcopy(zVal)
	zTargetMin = deepcopy(zVal)
	for idx in range(len(zVal)):
		zTargetMax[idx] = np.max(zVal[idx:])
		zTargetMin[idx] = np.min(zVal[:idx+1])

	zTargetMax = np.where(zTargetMax > zVal+zDiffAvg, zTargetMax, zVal)
	zTargetMin = np.where(zTargetMin < zVal-zDiffAvg, zTargetMin, zVal)

	zTargMaxRatio = np.linspace(0.0, 1.0, len(zVal))
	zTargMaxRatio = zTargMaxRatio
	zTarget = zTargMaxRatio*zTargetMax + (1.0-zTargMaxRatio)*zTargetMin

	# zTarget = (zTargetMax+zTargetMin) / 2.0
	slopeDownForce[2] = (zTarget - zVal) * forceMag


	# zTarget = deepcopy(zVal)
	# for idx in range(len(zTarget)):
	# 	zTarget[idx] = (np.max(zVal[idx:]) + np.min(zVal[:idx+1])) / 2
	# slopeDownForce[2] = (zTarget - zVal) * forceMag



	# for idx in range(len(zDiff)-1, -1, -1):
	# 	if zDiff[idx] > 0.0 and startIdx == None:
	# 		startIdx = idx
	# 	if zDiff[idx] < 0.0 and startIdx:
	# 		zModIdx = np.arange(startIdx, idx)
	# 		zTargets = np.interp(
	# 			zModIdx,
	# 			[startIdx, idx],
	# 			[zVal[startIdx], zVal[idx]],
	# 		)
			
	# 		slopeDownForce[2, zModIdx] = zTargets - zVal[zModIdx] * forceMag
			
	# 		startIdx = None
	# 		print(f"{startIdx}, {idx}")

		
	# zIncIdx = np.where( > 0.0)[0]
	# zIncIdx = np.unique(np.concatenate([zIncIdx, zIncIdx+1]))
	# valleyStarts = np.where(np.diff(zIncIdx) > 1)[0]
	# valleyPoints = np.concatenate([[0], valleyStarts+1])
	
	# setZValues = np.interp(
	# 	zIncIdx,
	# 	zIncIdx[valleyPoints],
	# 	zVal[zIncIdx[valleyPoints]]
	# )
	# slopeDownForce[2, zIncIdx] = (setZValues - zVal[zIncIdx]) * forceMag
	return(slopeDownForce)

def subdividePath(path, only_return_new=False, neverSlopeUp=True):
	tck, u = splprep(path, s=0)
	pathInterp = (u[1:] + u[:-1]) / 2
	new_points = splev(pathInterp, tck)
	new_points = np.array(new_points, dtype=np.double)


	# Do not slope up ever
	if neverSlopeUp:
		whereZincreases = np.where(new_points[2] > path[2, :-1])[0]
		new_points[2, whereZincreases] = path[2, whereZincreases]

	if only_return_new:
		return np.array(new_points)
	else:
		allPoints = np.zeros((path.shape[0], path.shape[1] + new_points.shape[1]), dtype=np.double)
		allPoints[:, ::2] = path
		allPoints[:, 1::2] = new_points
		return allPoints

def smooth_array(data, window_size):
	"""
	Smooths a 1D numpy array using a moving average.
	
	Parameters:
	- data: 1D numpy array of floats
	- window_size: size of the moving average window
	
	Returns:
	- smoothed_data: 1D numpy array of smoothed values
	"""
	if window_size < 1:
		raise ValueError("Window size must be at least 1")
	
	if window_size > len(data):
		raise ValueError("Window size must be less than or equal to the length of the data array")
	
	# Create the window for moving average
	window = np.ones(int(window_size)) / float(window_size)
	
	# Apply convolution between the data and the window
	smoothed_data = np.convolve(data, window, 'valid')
	
	# Handle the edges by padding with the original data
	pad_left = data[:window_size//2]
	pad_right = data[-(window_size//2):] if window_size % 2 == 0 else data[-(window_size//2 + 1):]
	smoothed_data = np.concatenate((pad_left, smoothed_data, pad_right))
	
	return smoothed_data

def hamming_filter_1d(data, window_size):
    if window_size % 2 == 0:
        raise ValueError("Window size should be odd")
    
    # Create Hamming window
    hamming_window = np.hamming(window_size)
    
    # Normalize the window
    hamming_window /= np.sum(hamming_window)
    
    # Pad the input data
    pad_width = window_size // 2
    padded_data = np.pad(data, pad_width, mode='edge')
    
    # Apply the filter
    filtered_data = np.convolve(padded_data, hamming_window, mode='valid')
    
    return filtered_data

def max_by_absolute_value(array1, array2):
	result = np.where(np.abs(array1) > np.abs(array2), array1, array2)
	return result

def smoothByPrevN(inputArr, N):
	convArr = np.flip(
			np.convolve(
				np.flip(inputArr),
				np.ones(int(N)) / float(N),
				mode='valid'
			)
		)
	inputArr[N-1:] = convArr
	return(inputArr)

def smoothByNextN(inputArr, N):
	convArr = np.convolve(
				inputArr,
				np.ones(int(N)) / float(N),
				mode='valid'
			)
	inputArr[:-(N-1)] = convArr
	return(inputArr)

def calculatePathRotations(path, screwJoinAngle=None):
	# Calculate angles
	baseAngles = np.arctan2(path[1, 2:] - path[1, :-2], path[0, 2:] - path[0, :-2])
	angles = np.arctan2(np.diff(path[1]), np.diff(path[0]))
	changeInAngle = np.diff(angles)
	changeInAngle[changeInAngle > np.pi] -= 2*np.pi
	changeInAngle[changeInAngle < -np.pi] += 2*np.pi

	# Calculate slopes
	pointDists = magnitude((path[:2, 2:] - path[:2, :-2]))
	pointSlopes = (path[2][2:] - path[2][:-2])/pointDists
	pointSlopesStandardized = (PT_DROP - np.diff(path[2])) / PT_SPACING

	# Convert slope at each point into a multiplier
	pointSlopes *= -3
	pointSlopes -= np.min(pointSlopes)
	slopeMagAtPoint = pointSlopes / np.average(pointSlopes)
	slopeMagAtPoint += 0.5
	# slopeMagAtPoint[slopeMagAtPoint < 1.0] = 1.0


	# Set minimum slope factor decay point to point
	slopeMagAtPoint = np.where(slopeMagAtPoint > 2.5, 2.5, slopeMagAtPoint)
	# maxSlope = 0.96
	maxSlopeDecay = 0.05
	slopeConv = deepcopy(slopeMagAtPoint)
	for idx in range(1, slopeConv.shape[0]):
		if slopeConv[idx] < slopeConv[idx-1] - maxSlopeDecay:
			slopeConv[idx] = slopeConv[idx-1] - maxSlopeDecay

	slopeConv = np.clip(slopeConv, 0.0, np.inf)

	# Smooth out slope
	slopeConv = smoothByPrevN(slopeConv, 3)

	# Get initial, raw tilt
	tilt = -changeInAngle*3

	# # Set beginning and ending points to flat
	# tilt[:LOCKED_PT_CNT-3] = 0.0
	# tilt[-LOCKED_PT_CNT+3:] = 0.0

	# Zero tilt of back and forth motion
	if False:
		reversePointDist = 1
		positiveTurnPoints = np.zeros_like(changeInAngle, dtype=np.int16)
		positiveTurnPoints[changeInAngle > 0.0] = 1
		# reversingPoints = np.where((positiveTurnPoints[reversePointDist*2:] == positiveTurnPoints[:-reversePointDist*2]) & (positiveTurnPoints[:-reversePointDist*2] != positiveTurnPoints[reversePointDist:-reversePointDist]))
		reversingPoints = np.where((positiveTurnPoints[reversePointDist*2:] != positiveTurnPoints[reversePointDist:-reversePointDist]) | (positiveTurnPoints[:-reversePointDist*2] != positiveTurnPoints[reversePointDist:-reversePointDist]))
		
		tilt[reversingPoints] = 0.0

	preClipTilt = deepcopy(tilt)


	# Limit max rotation a little to prevent hard turns from blowing out resolution
	PRE_SMOOTH_MAX_TILT = TRACK_MAX_TILT*1.0
	tilt = np.clip(tilt, -PRE_SMOOTH_MAX_TILT, PRE_SMOOTH_MAX_TILT)


	# Smooth tilts
	SMOOTH_CNT = 1
	SMOOTH_REP = 10
	currTilts = deepcopy(tilt)
	for ii in range(SMOOTH_REP):
		# currTilts = smooth_array(currTilts, SMOOTH_CNT)
		smoothTilts = np.convolve(currTilts, np.ones(int(SMOOTH_CNT*2+1)) / float(SMOOTH_CNT*2+1), mode='valid')
		currTilts[SMOOTH_CNT:-SMOOTH_CNT] = smoothTilts
		# currTilts = max_by_absolute_value(currTilts, tilt)

		currTilts[:2] = 0
		currTilts[-2:] = 0

	# Find distance to path orientation flipping
	turnSign = -np.ones_like(changeInAngle, dtype=np.int8)
	turnSign[changeInAngle > 0.0] = 1
	flipDist = distance_to_sign_change(turnSign)
	
	flipDistMag = flipDist/ 3
	flipDistMag[flipDistMag > 1.0] = 1.0
	

	# # plt.plot(turnSign)
	# plt.plot(flipDistMag)
	# plt.plot(changeInAngle)
	# plt.show()
	# exit()


	# Multiply by slopeConv
	# currTilts = smoothByNextN(deepcopy(currTilts), 3)*slopeConv*2
	
	currTilts = smoothByNextN(deepcopy(currTilts*flipDistMag), 3)*slopeConv*2

	# Limit max rotation
	currTilts = np.clip(currTilts, -TRACK_MAX_TILT, TRACK_MAX_TILT)

	# Reduce tilts for final points
	currTilts *= np.interp(
		np.arange(currTilts.shape[0]),
		[0.0, currTilts.shape[0]-END_RAIL_PTS, currTilts.shape[0]-END_RAIL_PTS+END_RAIL_TRANSITION],
		[1.0, 1.0, 0.0]
	)

	# # Reduce initial tilts
	# ZERO_PTS = LOCKED_PT_CNT*2
	# INIT_PTS = 10
	# currTilts[:ZERO_PTS] = 0.0
	# currTilts[ZERO_PTS:ZERO_PTS+INIT_PTS] = currTilts[ZERO_PTS:ZERO_PTS+INIT_PTS]*np.linspace(0.0, 1.0, INIT_PTS)

	# currTilts[-LOCKED_PT_CNT:] = 0.0
	# currTilts[-LOCKED_PT_CNT*2:-LOCKED_PT_CNT] *= np.linspace(1.0, 0.0, LOCKED_PT_CNT)

	if False:
		plt.plot(currTilts, label="currTilts")
		plt.plot(tilt, label="tilt")
		plt.plot(slopeConv, label="slopeConv")
		plt.plot(-changeInAngle*2, label="changeInAngle")
		plt.plot(slopeMagAtPoint, label="slopeMagAtPoint")
		plt.plot([0, len(tilt)], [0, 0])
		plt.plot(preClipTilt, label="preClipTilt")
		plt.legend()
		plt.show()
		# exit()

	# Set output array
	rotations = np.zeros_like(path)[:2]
	rotations[0, 1:-1] = baseAngles
	rotations[0, 0] = angles[0]
	rotations[0, -1] = angles[-1]
	rotations[1, 1:-1] = currTilts

	# # Set initial track to be flat
	# if screwJoinAngle != None:
	# 	forceRotMag = np.linspace(0.0, 1.0, LOCKED_PT_CNT)
	# 	# forceRotMag = np.interp(np.linspace(0.1, 1.0, LOCKED_PT_CNT), [0.0, 0.7], [0.0, 1.0])

	# 	rotations[0, :int(LOCKED_PT_CNT)] = -forceRotMag
	# 	rotations[1, :LOCKED_PT_CNT] = screwJoinAngle*rotations[1, :LOCKED_PT_CNT]

	# 	forceRotMag = np.flip(forceRotMag)

	# 	rotations[0, -int(LOCKED_PT_CNT):] = -forceRotMag
	# 	rotations[1, -LOCKED_PT_CNT:] = screwJoinAngle*rotations[1, -LOCKED_PT_CNT:]



	# rotations[1, :LOCKED_PT_CNT*2] = 0.0
	
	

	return rotations

def redistributePathByForce(path, sumForce):
	# Resample path increasing number of points in high force areas to attempt to relieve knots
	forceMag = magnitude(sumForce)

	forceMag += 3*np.average(forceMag) # add baseline
	interpPosition = np.cumsum(forceMag)
	
	ptCnt = path.shape[1]
	newPath = np.zeros_like(path)


	for idx in range(3):
		newPath[idx] = np.interp(
			np.linspace(0, interpPosition[-1], ptCnt), 
			interpPosition, 
			path[idx]
		)

	return(newPath)

def create_weighted_kernel(size, sigma):
	"""
	Creates a weighted kernel where the weights decrease with distance from the center.
	
	Parameters:
	- size: size of the kernel (must be an odd number)
	- sigma: standard deviation for the Gaussian function
	
	Returns:
	- kernel: 1D numpy array representing the weighted kernel
	"""
	# Ensure the size is odd to have a central element
	if size % 2 == 0:
		raise ValueError("Size must be an odd number")
	
	# Create an array of distances from the center
	distances = np.arange(-size // 2 + 1, size // 2 + 1)
	
	# Create a Gaussian kernel
	kernel = np.exp(-distances**2 / (2 * sigma**2))
	
	# Normalize the kernel to make the sum of weights equal to 1
	kernel /= kernel.sum()
	
	return kernel

def weighted_average_convolution(data, kernel_size=5, sigma=1.0):
	"""
	Applies a weighted average convolution to a 1D numpy array.
	
	Parameters:
	- data: 1D numpy array of floats
	- kernel_size: size of the weighted kernel (must be an odd number)
	- sigma: standard deviation for the Gaussian function used in the kernel
	
	Returns:
	- result: 1D numpy array of convolved values
	"""
	kernel = create_weighted_kernel(kernel_size, sigma)
	
	# Apply convolution with the 'same' mode to keep the array size the same
	result = np.convolve(data, kernel, mode='same')
	
	return result

def addToPathAndSums(force, path, forceSum, moveMultMag):
	path += force*moveMultMag
	forceSum += force

import threading
import matplotlib.pyplot as plt
from queue import Queue
import time

# Plot imported dictionaries
def data_processor_and_plotter(data_queue):
	data_dict = {}

	plt.ion()  # Turn on interactive mode
	fig, ax = plt.subplots()

	while True:
		while not data_queue.empty():
			new_data = data_queue.get()
			for tag, value in new_data.items():
				if tag not in data_dict:
					data_dict[tag] = []
				data_dict[tag].append(value)
		
		ax.clear()
		for tag, values in data_dict.items():
			ax.plot(values, label=tag)
		
		ax.legend()
		plt.draw()
		# plt.ylim(0, 1)
		plt.pause(1)  # Pause to update the plot


def plot_paths_real_time(data_queue):
	# fig, ax = plt.subplots()
	ax = plt.figure().add_subplot(projection='3d')
	ax.set_xlabel('X')
	ax.set_ylabel('Y')
	ax.set_zlabel('Z')
	
	plt.ion()  # Turn on interactive mode

	while True:
		pathList = None
		while not data_queue.empty():
			pathList = data_queue.get()
		
		if pathList:
			plotPath(ax, pathList)
		plt.draw()
		plt.pause(1.0)  # Pause to update the plot

def plotPath(ax, pathList):
	ax.clear()
	
	for pathIdx in range(len(pathList)):
		path = pathList[pathIdx]
		bridgePoints = pathList[pathIdx]



		# ax.scatter(*centerPoints)

		ax.scatter(*path)

		# ax.scatter(*bridgePoints[:, 1::2], color='purple')
		ax.plot(*bridgePoints, alpha=0.5)

		# ax.scatter(*path[:, :LOCKED_PT_CNT], color='red')
		# ax.scatter(*path[:, -LOCKED_PT_CNT:], color='red')

		if False:
			offset = 1
			forceSet = np.zeros_like(path)
			
			testPath = path
			if True:
				rad, cv1, cv2, cv3 = approximatePathCurvatureXY(testPath, offset=offset)
				cv1 = np.array([*cv1, np.zeros(cv1.shape[1])])
				cv2 = np.array([*cv2, np.zeros(cv2.shape[1])])
				cv3 = np.array([*cv3, np.zeros(cv3.shape[1])])
			else:
				rad, cv1, cv2, cv3 = approximatePathCurvature(testPath, offset=offset)

			rad = 5
			cv1 *= rad
			cv2 *= rad
			cv3 *= rad
			
			for ii in range(cv1.shape[1]):
				# plt.plot(*zip(testPath[:, ii], testPath[:, ii] + cv1[:, ii]), color='purple')
				plt.plot(*zip(testPath[:, ii+offset], testPath[:, ii+offset] + cv2[:, ii]), color='red')
				# plt.plot(*zip(testPath[:, ii+2*offset], testPath[:, ii+2*offset] + cv3[:, ii]), color='orange')


			
		if False:
			offset = 1
			forceSet = tempCurvatureCalc(path)

			for idx in range(len(path[0])):
				pt = path[:, idx]
				vect = forceSet[:, idx]
				ax.plot(*np.swapaxes([pt, vect+pt], 0, 1), color='black')


		# # visForce = normalizePathDists(path,  PT_SPACING, 1.0, maxForce=10.0)*5
		# visForce = correctPathAngle(path, 2.5, 3.14, 1.5, diffPointOffsetCnt=2)
		# for idx in range(path.shape[1]):
		# 	pt = path[:, idx]
		# 	vect = visForce[:, idx]
		# 	ax.plot(*np.swapaxes([pt, vect+pt], 0, 1), color='orange')
		
	ax.set_xlim(0, SIZE_X)
	ax.set_ylim(0, SIZE_Y)
	ax.set_zlim(0, SIZE_Z)
	ax.set_aspect('equal', adjustable='box')

def loadChangesToQueue(file_path, pathQueue):
	# Load initial value
	pathList = pkl.load(open(file_path, 'rb'))
	pathQueue.put(pathList)

	from watchdog.observers import Observer
	from watchdog.events import FileSystemEventHandler

	class FileChangeHandler(FileSystemEventHandler):
		def __init__(self, file_path, pathQueue):
			self.file_path = file_path
			self.pathQueue = pathQueue

		def on_modified(self, event):
			if event.src_path == self.file_path and os.path.exists(self.file_path):
				try:
					pathList = pkl.load(open(self.file_path, 'rb'))
					self.pathQueue.put(pathList)
				except:
					print(f"Failed to load path")

	event_handler = FileChangeHandler(file_path, pathQueue)

	observer = Observer()
	observer.schedule(event_handler, path=file_path, recursive=False)
	observer.start()


	try:
		while True:
			time.sleep(1.0)
	except KeyboardInterrupt:
		observer.stop()
	observer.join()


# If main, plot in separate thread
if __name__ == '__main__':
	file_path = WORKING_DIR+'path.pkl'

	# Start the asynchronous data polling thread
	pathQueue = Queue()
	pathPlottingThread = threading.Thread(target=loadChangesToQueue, args=(file_path, pathQueue))
	pathPlottingThread.daemon = True
	pathPlottingThread.start()

	# Plot in real time on main thread
	plot_paths_real_time(pathQueue)



	# Test path curvature
	if False:
		testPath = np.array([
			[0, 1, 2, 3, 4, 4.5],
			[1, 0, 1, 0, 0, -0.75]
		])

		offset = 2
		rad, cv1, cv2, cv3 = approximatePathCurvatureXY(testPath, offset=offset)
		print(f"rad:{rad}")
		print(f"cv1:{cv1}")
		print(f"cv2:{cv2}")
		print(f"cv3:{cv3}")


		plt.scatter(*testPath)
		# plt.scatter(*(testPath[:, 1:-1] + cent))
		for ii in range(rad.shape[0]):
			plt.plot(*zip(testPath[:, ii], testPath[:, ii] + cv1[:, ii]), color='purple')
			plt.plot(*zip(testPath[:, ii+offset], testPath[:, ii+offset] + cv2[:, ii]), color='red')
			plt.plot(*zip(testPath[:, ii+2*offset], testPath[:, ii+2*offset] + cv3[:, ii]), color='orange')


			angles = np.linspace(0.0, np.pi*2, 30)
			centerPoint = testPath[:, ii+offset] + cv2[:, ii]*rad[ii]

			circlePts = np.array([
				np.cos(angles)*rad[ii] + centerPoint[0],
				np.sin(angles)*rad[ii] + centerPoint[1],
			])
			plt.plot(*circlePts, color='black')

		plt.xlim(-5, 10)
		plt.ylim(-5, 10)
		plt.show()
