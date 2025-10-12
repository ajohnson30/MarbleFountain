import numpy as np
import sys

GENERATE_SUPPORTS = True # Actually generate supports
HOLLOW_SUPPORTS = False # Hollow out supports (to put lights in)
LED_CUTOUTS = False # Add cutouts to put LEDs in
CONNECT_LIFTS = True
SOLID_WALL_BETWEEN_LIFTS = False
LOAD_EXISTING_PATH = True
REALTIME_PLOTTING_FORCEMAGS = False # Plot path forces in real time
SUPPORT_VIS = True # Output support gen visualization
GLASS_MARBLE_14mm = False
MIRROR_PATHS = False
SMOOTH_SUPPORTS = True # Smooth support geometry to 



# Overall size of box to generate path in
SIZE_X = 160
SIZE_Y = 100
SIZE_Z = 180
PT_DROP = 0.8    # target z drop per pt
PATH_COUNT = 3 # Numer of paths to generate
SCREW_RAD = 12 # Center of rotation to center of marble on track
SCREW_PITCH = 40 # mm per rev
LIFT_SUPPORT_PTS = 21
# SOLID_WALL_BETWEEN_LIFTS = True
RANDOM_CNT = 6  # How many random points to generate if LESS_RANDOM_INIT_PATH


# # SIZE_X = 315
# # SIZE_Y = 185
# # SIZE_Z = 270

SIZE_X = 350 - 20
SIZE_Y = 200 - 25
SIZE_Z = 353.28 - 50
PT_DROP = 0.55    # target z drop per pt
PATH_COUNT = 8 # Number of paths to generate
SCREW_RAD = 18 # Center of rotation to center of marble on track
SCREW_PITCH = 24 # mm per rev
LIFT_SUPPORT_PTS = 51
MIRROR_PATHS = False
RANDOM_CNT = 15  # How many random points to generate if LESS_RANDOM_INIT_PATH

WORKING_DIR = 'FinalDemo/'
PT_DROP = 0.85    # target z drop per pt

# PT_DROP = 0.8    # target z drop per pt
# PATH_COUNT = 5
# WORKING_DIR = 'proc/Print39/'
# MIRROR_PATHS = True



# PT_DROP = 0.8    # target z drop per pt
# PATH_COUNT = 8 # Numer of paths to generate
# GLASS_MARBLE_14mm = True
# SIZE_X = 320 - 20
# SIZE_Y = 200 - 40
# SIZE_Z = 353.28 - 65
# WORKING_DIR = 'proc/Print29_GlassMarble/'



if len(sys.argv) > 1 and sys.argv[1] != '-reset':
    WORKING_DIR = sys.argv[1]


BASE_OF_MODEL = -12 # Offset from 0 in Z to print main body off of
BASE_THICKNESS = 6 # Offset from 0 in Z to print main body off of

PT_SPACING = 6 # distance from one point to the next

INITIAL_POINT_MULT_SLOPE = 5.0*PT_DROP
POINT_COUNT = int(np.floor(SIZE_Z / PT_DROP)) # Total number of path points
if POINT_COUNT%2 == 0: POINT_COUNT += 1

if MIRROR_PATHS:
    SIZE_X = SIZE_Y


# Path gen optimization
PATH_ITERS = 10000 # Number of iterations to optimize too
RESAMPLE_AT = [] # Resample the path to alleviate knots at this number of iterations
APPLY_FORCES_SEPARATELY = True
SET_ITERATION_MOVE_DISTS = False # Move all points by same distance which gradually decreases (instead of by force)
LESS_RANDOM_INIT_PATH = True # Generate initial paths by interpolating between a few random paths (instead of randomizing every point)

# Path randomization
#   Based on max force mag, calculates temperature and temp decay
#   A positive temp indicates mag of random noise added
#   A negative temp indicates magnitude of force updates, with -10 being 0
#   Pairs are (max force mag, noise setting, temp decay)
DO_DYNAMIC_TEMPERATURE = True
TEMPERATURE_HISTORY_LEN = 400
TEMPERATURE_FAILURE_BOOST = 20.0
PATH_RANDOMIZATION_FUNC = np.swapaxes([
    [10.0, -10.0, 0.1],
    [13.0, 0.0, 0.05],
    [13.1, 1.0, 0.1],
    [17.0, 2.0, 0.1],
    [40.0, 15.0, 0.2],
    [200.0, 40.0, 0.5], # Max noise of 20
], 0, 1)


PATH_RANDOMIZATION_FUNC = np.swapaxes([
    [9.0, -10.0, 0.1],
    [12.0, 0.0, 0.05],
    [12.1, 1.0, 0.1],
    [16.0, 2.0, 0.1],
    [40.0, 15.0, 0.2],
    [200.0, 40.0, 0.5], # Max noise of 20
], 0, 1)

# PATH_RANDOMIZATION_FUNC[0] *= 1.6
# PATH_RANDOMIZATION_FUNC[0, 0] = 14

LOCKED_PT_CNT = 5 # Points locked in a straight line as part of the initial path

# Track defs
MARBLE_RAD = 6.3/2 # Radius of marble
TRACK_RAD = 1.0 # Radius of standard marble support section

TRACK_CONTACT_ANGLE = np.pi/5 # Angle between path and contact points4
TRACK_MAX_TILT = np.pi/5 # Max angle of tilt for track

if GLASS_MARBLE_14mm:
    # PT_SPACING = 10.0
    MARBLE_RAD = 7.0
    TRACK_RAD = 1.5
    TRACK_SUPPORT_RAD = 1.5
    PATH_COUNT = 4
    PT_DROP = 0.8
    BASE_OF_MODEL = -25

    TRACK_MAX_TILT = np.pi/10 # Max angle of tilt for track


END_RAIL_PTS = 36 # How many points to start transitioning to final track
END_RAIL_TRANSITION = 8 # How many points take to transition to the final track
END_RAIL_CONTACT_ANGLE = np.pi/5 # Contact angle for base rail at the end of the path
END_RAIL_GUIDE_CONTACT_ANGLE = -np.pi/4
END_RAIL_GUIDE_TILT = np.pi/4 # What angle to tilt the guide rail geometry to for the end of the path
END_RAIL_GUIDE_MARGIN = 1.2 # Additional tolerancing spacing of guide rails 
END_RAIL_CONTACT_INIT_MARGIN = 0.5 # Additional spacing at start of initial base rail to help smooth transition


SMOOTH_TILT_CNT_A = 2 # How many points to smooth each rotation points' tilt by
SMOOTH_TILT_CNT_B = 3 # And then we do it again

# Support generation constants
TRACK_SUPPORT_RAD = 1.0 # Initial radius of track support
TRACK_SUPPORT_MAX_RAD = 2.2 # Maximum support radius
SUPPORT_LAYER_HEIGHT = 0.5 # Layer height
MAX_PARTICLE_VEL = SUPPORT_LAYER_HEIGHT*2.0 # Maximum XY motion between each layer
MAX_PARTICLE_ACC = SUPPORT_LAYER_HEIGHT*0.3 # Maximum XY acceleration between each layer (except in case of emergency to avoid collision)
MERGE_RAD = MAX_PARTICLE_VEL*MAX_PARTICLE_VEL / 2 # Radius to merge points beneath, set to 1/10 to encourage supports to swirl
MERGE_SMOOTH_PTS = 8 # How many points to start resizing column before join

PARTICLE_DRAG = 0.8 # Fraction of velocity retained across frames
SUPPORT_ATTRACTION_CONSTANT = 80.0 # Constant multiplier for attraction force between particles
SUPPORT_MAX_ATTRACTION_DIST = 50.0 # DISABLED max attraction distance
SUPPORT_BOUNDARY_FORCE_MAG = 20.0 # Force of boundary limitation, in force/mm

# Support avoidPt constants
PEAK_REPULSION_MAG = 5.0 # Maximum force generated by point avoidance
Z_DIFF_MIN = MARBLE_RAD*3 # Z difference of min repulsion force
Z_DIFF_MAX = MARBLE_RAD*6 # Z difference of max repulsion force
POS_DIFF_MIN = MARBLE_RAD*1.5 # min XY diff of repulsion force
POS_DIFF_MAX = MARBLE_RAD*3.0 # max XY diff of repulsion force
PATH_REPEL_DROP = MARBLE_RAD*2.0 # Drop points this far in Z from center of point

PULL_TO_CENTER_MAG = 0.2 # Magnitude of force pulling points to target radius
PULL_TO_CENTER_MAXDIST = 10.0 # Distance at which to cap pull to target rad



SUPPORT_SIZE_INTERP = np.swapaxes([
    [1, 1.2],
    [2, 1.8],
    [7, 2.5],
    [20, 3.0]
], 0, 1)
SUPPORT_HOLLOW_INTERP = np.swapaxes([
    [1, 0.0],
    [2, 1.0],
    [7, 1.7],
    [20, 2.2]
], 0, 1)
SUPPORT_BASE_RAD = 3.5

if not HOLLOW_SUPPORTS:
    SUPPORT_SIZE_INTERP = np.swapaxes([
        [1, 1.0],
        [3, 2.0],
        [15, 3.0]
    ], 0, 1)

LIFT_SUPPORT_DIST = TRACK_RAD*3
LIFT_SUPPORT_CROSSES = 16
LIFT_SUPPORT_SUBDIV = 10

# How many sides for each circle
UNIVERSAL_FN = 6
HIGHER_RES_FN = 10

UNIVERSAL_FN = 18
HIGHER_RES_FN = 50

# Screw lift
SCREW_RESOLUTION = 30 # pts per rev
SCREW_SUPPORT_GROUPING = (4, 9)
SCREW_OUTER_TRACK_DIST = -0.25 # How far out to place the bottom rail
SCREW_VERT_RAIL_MARGIN = 0.25 # Margin between bottom rail and vertical lift supports
SCREW_TOP_PUSH_PTS = 15 # How many points at the top of the screw to push towards the outside


if GLASS_MARBLE_14mm:
    SCREW_RAD = 16


# Entry and exit connections
END_PATH_OFFSET = MARBLE_RAD * (1 + np.sin(TRACK_CONTACT_ANGLE))

MOTOR_TYPE = 'SMALL_DC'
MOTOR_TYPE = 'NEMA17'

SCREW_POS = np.array([SIZE_X/2, SIZE_Y/2, 0.0])


BOUNDING_BOX = np.array([SIZE_X, SIZE_Y, SIZE_Z])
OUTPUT_BASE_RAD = np.min([SIZE_X, SIZE_Y]) / 2


ABSOLUTE_MIN_PATH_DIST = (MARBLE_RAD + TRACK_RAD)
ABSOLUTE_MIN_PT_DIST = np.sqrt(np.square(2*ABSOLUTE_MIN_PATH_DIST) + np.square(PT_SPACING/2))
