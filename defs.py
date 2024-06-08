import numpy as np

# Big params
SIZE_X = 120
SIZE_Y = 195
SIZE_Z = 100

PT_SPACING = 6 # distance from one point to the next

PT_DROP = 1.5 # target z drop per pt
POINT_COUNT = int(np.floor(SIZE_Z / PT_DROP))

PATH_COUNT = 4

# Path gen optimization
PATH_ITERS = 150
RESAMPLE_AT = [75]
SET_ITERATION_MOVE_DISTS = False
LESS_RANDOM_INIT_PATH = True
RANDOM_CNT = 10

INIT_PATH_PTS = POINT_COUNT
LOCKED_PT_CNT = 5 # Points locked in a straight line as part of the initial path

# Track defs
MARBLE_RAD = 6.3/2
TRACK_RAD = 0.75 # Radius of standard marble support section
TRACK_CONTACT_ANGLE = np.pi/4

TRACK_MAX_TILT = np.pi - TRACK_CONTACT_ANGLE

TRACK_SUPPORT_RAD = 0.75


UNIVERSAL_FN = 10

# Screw lift
SCREW_RAD = 16 # Center of rotation to center of marble on track
SCREW_PITCH = 16 # mm per rev
SCREW_RESOLUTION = 30 # pts per rev
SCREW_SUPPORT_GROUPING = 5
SCREW_OUTER_TRACK_DIST = 0.0 # How far out to place the bottom rail
SCREW_TOP_PUSH_PTS = 10 # How many points at the top of the screw to push towards the outside

WORKING_DIR = 'proc/Test/'




BOUNDING_BOX = np.array([SIZE_X, SIZE_Y, SIZE_Z])

ABSOLUTE_MIN_PATH_DIST = (MARBLE_RAD + TRACK_RAD)
ABSOLUTE_MIN_PT_DIST = np.sqrt(np.square(2*ABSOLUTE_MIN_PATH_DIST) + np.square(PT_SPACING/2))
