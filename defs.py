import numpy as np

# Big params
SIZE_X = 120
SIZE_Y = 195
SIZE_Z = 100

PT_SPACING = 6 # distance from one point to the next

PT_DROP = 1 # target z drop per pt
POINT_COUNT = int(np.floor(SIZE_Z / PT_DROP))

PATH_ITERS = 110
PATH_COUNT = 3
INIT_PATH_PTS = POINT_COUNT

# Track defs
MARBLE_RAD = 6.3/2
TRACK_RAD = 0.75 # Radius of standard marble support section

# Screw lift
SCREW_RAD = 16 # Center of rotation to center of marble on track
SCREW_PITCH = 16 # mm per rev
SCREW_RESOLUTION = 30 # pts per rev
SCREW_OUTER_TRACK_DIST = 0.0 # How far out to place 

WORKING_DIR = 'proc/Test'

BOUNDING_BOX = np.array([SIZE_X, SIZE_Y, SIZE_Z])

ABSOLUTE_MIN_PATH_DIST = (MARBLE_RAD + TRACK_RAD)
ABSOLUTE_MIN_PT_DIST = np.sqrt(np.square(2*ABSOLUTE_MIN_PATH_DIST) + np.square(PT_SPACING/2))
