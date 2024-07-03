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
def randomPath(ptCnt, box):
	path = np.zeros((3, ptCnt), dtype=np.double)
	if LESS_RANDOM_INIT_PATH:
		# Sort of random points
		for idx in range(3):
			randPts = np.random.random(RANDOM_CNT+2)
			randPts[0] = 0.5
			randPts[-1] = 0.5
			path[idx] = np.interp(np.linspace(0, RANDOM_CNT+2, ptCnt), np.arange(RANDOM_CNT+2), randPts)*box[idx]
	else:
		# Fully random points
		path[:3] = np.random.random(path[:3].shape)
		path[0] *= box[0]
		path[1] *= box[1]
		path[2] *= box[2]
	
	return(path)

def getPathAnchorAngle(pathIdx):
	angle = np.pi*2*pathIdx/PATH_COUNT
	if PATH_COUNT%2 == 0: angle += np.pi/(PATH_COUNT)
	return angle

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
def normalizePathDists(path, targDist, forcePerDist, maxForce = 5.0, pointOffset = 1):
	pathDiffs = path[:, pointOffset:] - path[:, :-pointOffset]
	pathDists = magnitude(pathDiffs)

	pathNorms = pathDiffs / pathDists

	forceMags = (targDist - pathDists) * forcePerDist / 2

	forceMags = np.clip(forceMags, -maxForce, maxForce)
	# forceMags = np.max([forceMags, np.zeros_like(forceMags)-10], axis=0)

	outForces = np.zeros_like(path)
	outForces[:, :-pointOffset] -= forceMags * pathNorms
	outForces[:, pointOffset:] += forceMags * pathNorms

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

# Smooth out change in slope
def correctSlopeChange(path, forceMag = 1.0, slopeErrMag=0.2):
	outForces = np.zeros_like(path)

	zDiffs = np.diff(path[2])
	xyDist = magnitude(np.diff(path[:2], axis=1))
	slope = zDiffs/xyDist

	correctSlope = (PT_DROP / PT_SPACING)
	
	slopeErr = correctSlope - slope
	outForces[2, 1:] -= slopeErr*slopeErrMag/2
	outForces[2, :-1] += slopeErr*slopeErrMag/2

	maxSlopeDelta = correctSlope
	maxSlopeDeltaCap = maxSlopeDelta*2
	slopeDiff = np.diff(slope)
	
	slopeDiffErrMag = np.interp(
		slopeDiff,
		[-maxSlopeDeltaCap, -maxSlopeDelta, maxSlopeDelta, maxSlopeDeltaCap],
		[-1.0, 0.0, 0.0, 1.0]
	)
	outForces[2, 1:-1] = slopeDiffErrMag * forceMag

	return(outForces)

def preventUphillMotion(path, forceMag = 0.1):
	slopeDownForce = np.zeros_like(path)
	zVal = path[2]
	zDiff = np.diff(zVal)
	startIdx = None

	zTargetMax = deepcopy(zVal)
	zTargetMin = deepcopy(zVal)
	for idx in range(len(zVal)):
		zTargetMax[idx] = np.max(zVal[idx:])
		zTargetMin[idx] = np.min(zVal[:idx+1])
	zTargMaxRatio = np.linspace(0.0, 1.0, len(zVal))
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

def calculatePathRotations(path, diffPointOffsetCnt=2):
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
	pointSlopes *= -1
	pointSlopes -= np.min(pointSlopes)
	slopeMagAtPoint = pointSlopes / np.average(pointSlopes)
	slopeMagAtPoint += 0.5
	slopeMagAtPoint[slopeMagAtPoint < 1.0] = 1.0


	# Set minimum slope factor decay point to point
	maxSlope = 0.96
	slopeConv = deepcopy(slopeMagAtPoint)
	for idx in range(1, slopeConv.shape[0]):
		if slopeConv[idx] < slopeConv[idx-1]*maxSlope:
			slopeConv[idx] = slopeConv[idx-1]*maxSlope

	# Smooth out slope
	slopeConv = smoothByPrevN(slopeConv, 3)

	# Get initial, raw tilt
	tilt = -changeInAngle

	# Set beginning and ending points to flat
	tilt[:LOCKED_PT_CNT*3] = 0.0
	tilt[-LOCKED_PT_CNT:] = 0.0

	# Zero tilt of back and forth motion
	reversePointDist = 3
	positiveTurnPoints = np.zeros_like(changeInAngle, dtype=np.int16)
	positiveTurnPoints[changeInAngle > 0.0] = 1
	# reversingPoints = np.where((positiveTurnPoints[reversePointDist*2:] == positiveTurnPoints[:-reversePointDist*2]) & (positiveTurnPoints[:-reversePointDist*2] != positiveTurnPoints[reversePointDist:-reversePointDist]))
	reversingPoints = np.where((positiveTurnPoints[reversePointDist*2:] != positiveTurnPoints[reversePointDist:-reversePointDist]) | (positiveTurnPoints[:-reversePointDist*2] != positiveTurnPoints[reversePointDist:-reversePointDist]))
	
	tilt[reversingPoints] = 0.0

	preClipTilt = deepcopy(tilt)

	# Multiply by slopeConv
	tilt = smoothByNextN(deepcopy(tilt), 5)*slopeConv*3
	
	# Limit max rotation
	tilt = np.clip(tilt, -TRACK_MAX_TILT, TRACK_MAX_TILT)

	# Smooth tilts
	SMOOTH_CNT = 1
	SMOOTH_REP = 10
	currTilts = deepcopy(tilt)
	for ii in range(SMOOTH_REP):
		# currTilts = smooth_array(currTilts, SMOOTH_CNT)
		smoothTilts = np.convolve(currTilts, np.ones(int(SMOOTH_CNT*2+1)) / float(SMOOTH_CNT*2+1), mode='valid')
		currTilts[SMOOTH_CNT:-SMOOTH_CNT] = smoothTilts
		# currTilts = max_by_absolute_value(currTilts, tilt)



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

	# Set initial track to be flat
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
		while not data_queue.empty():
			pathList = data_queue.get()
			plotPath(ax, pathList)
		plt.draw()
		plt.pause(1.0)  # Pause to update the plot

def plotPath(ax, pathList):
	ax.clear()
	
	for pathIdx in range(len(pathList)):
		path = pathList[pathIdx][:, ::2]
		bridgePoints = pathList[pathIdx]


		# ax.scatter(*centerPoints)

		ax.scatter(*path)

		# ax.scatter(*bridgePoints[:, 1::2], color='purple')
		ax.plot(*bridgePoints, alpha=0.5)

		# ax.scatter(*path[:, :LOCKED_PT_CNT], color='red')
		# ax.scatter(*path[:, -LOCKED_PT_CNT:], color='red')

		if True:
			forceSet = preventUphillMotion(path, 1.0)
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
				pathList = pkl.load(open(self.file_path, 'rb'))
				self.pathQueue.put(pathList)

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