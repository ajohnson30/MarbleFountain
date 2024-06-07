import numpy as np
from scipy.interpolate import splprep, splev
from scipy.spatial.distance import cdist
from scipy.linalg import norm
from random import random
from copy import deepcopy
import pickle as pkl


def getAng(pt1, pt2): return(np.arctan2(pt2[1] - pt1[1], pt2[0] - pt1[0]))

def angDiff(a1, a2):
	a = a2 - a1
	if a > np.pi: a -= 2*np.pi
	if a < -np.pi: a += 2*np.pi
	return a

# Generate random initial path
def randomPath(ptCnt, box):
	path = np.zeros((3, ptCnt), dtype=np.double)
	path[:3] = np.random.random(path[:3].shape)
	path[0] *= box[0]
	path[1] *= box[1]
	path[2] *= box[2]
	# path[2] = np.arange(0, -box[2], -box[2]/ptCnt)
	
	return(path)


# Pull towards bounding box
def pushTowardsBoundingBox(pts, box, forcePerDist, maxForcePerAxis, axCount = 2):
	outForces = np.zeros_like(pts)
	zeros = np.zeros_like(pts[0])
	
	boxSet = np.tile(box, (pts.shape[1], 1)).T
	outForces[np.where(pts > boxSet)] -= (pts - boxSet)[np.where(pts > boxSet)]
	outForces[np.where(pts < 0.0)] -= pts[np.where(pts < 0.0)]
	# for ax in range(axCount):
	# 	outForces[ax] -= np.min([pts[ax], zeros])
	# 	outForces[ax] += np.min([box[ax] - pts[ax], zeros])

	outForces *= forcePerDist
	outForces = np.clip(outForces, -maxForcePerAxis, maxForcePerAxis)
	return(outForces)

# Pull towards Z position
def pullTowardsTargetHeights(pts, zTargetPositions, forcePerDist, maxForce=5):
	outForces = np.zeros_like(pts)
	outForces[2] = forcePerDist * (zTargetPositions - pts[2])
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
def normalizePathDists(path, targDist, forcePerDist):
	pathDiffs = path[:, :-1] - path[:, 1:]
	pathDists = magnitude(pathDiffs)

	pathDists[np.where(pathDists == 0.0)] = 1.0
	pathNorms = pathDiffs / pathDists

	# np.linalg.norm()
	forceMags = forcePerDist * (targDist - pathDists)

	# forceMags = np.clip(forceMags, -forcePerDist/2, forcePerDist/2)
	# forceMags = np.max([forceMags, np.zeros_like(forceMags)-10], axis=0)

	outForces = np.zeros_like(path)
	outForces[:, :-1] += forceMags * pathNorms
	outForces[:, 1:] += forceMags * (-pathNorms)

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
		ptForceMags = (peakForce/cutOffDist) * np.max([cutOffDist - ptDists, np.zeros_like(ptDists)], axis=0)

		outForces[:, ptIdx] = np.sum(ptForceMags * (ptDiffs / ptDists), axis=1) 

	return outForces

# Limit path angle
def correctPathAngle(path, minAng, maxAng, forcePerRad, flatten=True):
	if flatten:
		path = deepcopy(path)
		path[2] = 0

	# Calculate vectors and normals to preceding and succeeding point
	pathDiffs = path[:, 1:] - path[:, :-1]
	nextPtVect = pathDiffs[:, 1:]
	prevPtVect = -pathDiffs[:, :-1]

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
	outForces[:, 1:-1] -= forceMags*forceNormalVect	
	
	
	# # Apply inline force on each adjacent particle
	# outForces[:, :-2] += forceMags*prevNorm
	# outForces[:, 2:]  += forceMags*nextNorm

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
	outForces[:, 2:]  += forceMags * np.cross(nextNorm, crossProdNorm, axis=0)
	outForces[:, :-2] -= forceMags * np.cross(prevNorm, crossProdNorm, axis=0)

	return outForces


def subdividePath(path, only_return_new=False):
	tck, u = splprep(path, s=0)
	pathInterp = (u[1:] + u[:-1]) / 2
	new_points = splev(pathInterp, tck)
	new_points = np.array(new_points, dtype=np.double)

	if only_return_new:
		return np.array(new_points)
	else:
		allPoints = np.zeros((path.shape[0], path.shape[1] + new_points.shape[1]), dtype=np.double)
		allPoints[:, ::2] = path
		allPoints[:, 1::2] = new_points
		return allPoints


def calculatePathRotations(path):	
	baseAngles = np.arctan2(path[1, 2:] - path[1, :-2], path[0, 2:] - path[0, :-2])
	angles = np.arctan2(np.diff(path[1]), np.diff(path[0]))
	changeInAngle = np.diff(angles)

	rotations = np.zeros_like(path)[:2]
	rotations[0, 1:-1] = baseAngles
	rotations[0, 0] = angles[0]
	rotations[0, -1] = angles[-1]
	rotations[1, 1:-1] = -changeInAngle

	return rotations
