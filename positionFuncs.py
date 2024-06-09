from copy import deepcopy
from math import cos, sin
import math as m
import numpy as np



def doRotationMatrixes(inPts, rotations, transposed = False):
	A = rotations[0]
	B = rotations[1]
	C = rotations[2]

	rotationMags = np.array([        
		[ cos(B)*cos(C), sin(A)*sin(B)*cos(C)-cos(A)*sin(C), cos(A)*sin(B)*cos(C)+sin(A)*sin(C) ],
		[ cos(B)*sin(C), sin(A)*sin(B)*sin(C) +cos(A)*cos(C), cos(A)*sin(B)*sin(C) -sin(A)*cos(C) ],
		[ -sin(B), sin(A)*cos(B), cos(A)*cos(B) ],
	])
	
	if transposed: rotationMags = np.matrix.transpose(rotationMags)

	# if type(pos[0]) in (np.ndarray, list, set):
	xPts = sum([inPts[ii]*rotationMags[0][ii] for ii in range(3)])
	yPts = sum([inPts[ii]*rotationMags[1][ii] for ii in range(3)])
	zPts = sum([inPts[ii]*rotationMags[2][ii] for ii in range(3)])
	return([xPts, yPts, zPts])

	# else:
	# 	xPts = [inPts[ii]*rotationMags[0][ii] for ii in range(3)]
	# 	yPts = [inPts[ii]*rotationMags[1][ii] for ii in range(3)]
	# 	zPts = [inPts[ii]*rotationMags[2][ii] for ii in range(3)]
	# 	return([xPts, yPts, zPts])


	# for foo in rotationMags: print(foo)

def motionToZAng(inMotion):
	return np.arctan2(-sin(inMotion[4]), cos(inMotion[4])*cos(inMotion[5]))


def completeMotion(inPts, motion):
	offSets = motion[:3]
	rotations = motion[3:6]

	ptSet = doRotationMatrixes(inPts, rotations)

	for ii in range(3): 
		ptSet[ii] += offSets[ii]    
	
	return(ptSet)


def undoMotion(inPts, motion):
	offSets = motion[:3]
	rotations = motion[3:6]
	
	for ii in range(3): 
		inPts[ii] -= offSets[ii]    

	ptSet = doRotationMatrixes(inPts, rotations, transposed=True)
	
	return(ptSet)


# def applyMotion(pos, motion):
# 	mat = transformationMatrix(motion)
	
# 	return(
# 		pos[ii]*rotationMags[0][ii] for ii in range(3),
# 		pos[ii]*rotationMags[1][ii] for ii in range(3),
# 		pos[ii]*rotationMags[2][ii] for ii in range(3),
# 	)

def transformationMatrix(motion):
	A = motion[3]
	B = motion[4]
	C = motion[5]

	return(np.array([        
		[ cos(B)*cos(C), sin(A)*sin(B)*cos(C)-cos(A)*sin(C), cos(A)*sin(B)*cos(C)+sin(A)*sin(C), motion[0] ],
		[ cos(B)*sin(C), sin(A)*sin(B)*sin(C) +cos(A)*cos(C), cos(A)*sin(B)*sin(C) -sin(A)*cos(C), motion[1] ],
		[ -sin(B), sin(A)*cos(B), cos(A)*cos(B), motion[2] ],
		[0, 0, 0, 1]
	]))


def addMotions(motion1, motion2, transpose1 = False):
	if not transpose1: outMatrix = np.matmul( transformationMatrix(motion1), transformationMatrix(motion2) )
	else: outMatrix = np.matmul( np.linalg.inv(transformationMatrix(motion1)), transformationMatrix(motion2) )
	return(outMatrix)


def getMotionBetween(motion1, motion2):
	outMat = np.matmul( np.linalg.inv(transformationMatrix(motion2)), transformationMatrix(motion1)) 
	
	outMotion = [
		outMat[0][3],
		outMat[1][3],
		outMat[2][3],
		np.arctan2(outMat[2][1], outMat[2][2]),
		np.arctan2(-outMat[2][0], m.sqrt(pow(outMat[3][1], 2) + pow(outMat[2][2], 2))),
		np.arctan2(outMat[1][0], outMat[0][0]),
	]

	return(outMotion)


def normalizeMotion(inMotion):
	outMat = transformationMatrix(inMotion)
	
	outMotion = [
		outMat[0][3],
		outMat[1][3],
		outMat[2][3],
		np.arctan2(outMat[2][1], outMat[2][2]),
		np.arctan2(-outMat[2][0], m.sqrt(pow(outMat[3][1], 2) + pow(outMat[2][2], 2))),
		np.arctan2(outMat[1][0], outMat[0][0]),
	]

	return(outMotion)


def magnitude(inVals):
	return(m.sqrt(sum(pow(inVals,2))))

def getClosestPts(inVectors, inPts):
	inPts = np.column_stack(inPts)
	inVectors = np.column_stack(inVectors)

	tSet = np.sum(inPts*inVectors, axis=1) / np.sum(inVectors*inVectors, axis=1)
	# print(tSet)
	# print(tSet[:, None])
	outPts = inVectors * tSet[:, None] # Converts NP array to 2d with only one element in row
	# print(outPts)
	return(outPts)

def crossFixed(a:np.ndarray,b:np.ndarray)->np.ndarray: # Fix code unreachable error in some IDES
	return np.cross(a,b)

def ptVectDistSquared(pt, line):
	dVect = line - pt # Get vector from arbitrary point on line to target
	distance = np.sum(np.square(crossFixed(line, dVect)), axis=1)/np.sum(np.square(line), axis=1)
	return(distance)

def getError(inVectors, inPts, ptSize):
	inPts = np.column_stack(inPts)
	inVectors = np.column_stack(inVectors)

	distances = ptVectDistSquared(inVectors, inPts)
	# distances *= ptSize
	return( np.sum(distances) )

def testError(inVectors, inPts, motion, ptSize):
	adjPts = completeMotion(inPts, motion)
	sumError = getError(inVectors, adjPts, ptSize)
	return(sumError)






def set_axes_equal(ax):
	'''Make axes of 3D plot have equal scale so that spheres appear as spheres,
	cubes as cubes, etc..  This is one possible solution to Matplotlib's
	ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

	Input
	ax: a matplotlib axis, e.g., as output from plt.gca().
	'''

	x_limits = ax.get_xlim3d()
	y_limits = ax.get_ylim3d()
	z_limits = ax.get_zlim3d()

	x_range = abs(x_limits[1] - x_limits[0])
	x_middle = np.mean(x_limits)
	y_range = abs(y_limits[1] - y_limits[0])
	y_middle = np.mean(y_limits)
	z_range = abs(z_limits[1] - z_limits[0])
	z_middle = np.mean(z_limits)

	# The plot bounding box is a sphere in the sense of the infinity
	# norm, hence I call half the max range the plot radius.
	plot_radius = 0.5*max([x_range, y_range, z_range])

	ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
	ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
	ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])
	