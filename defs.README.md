Here are some other values to try in defs.py
Note: X,Y,Z should be sized based on your printers Build volume!!

Here's a small 3 track run:

# Overall size of box to generate path in
SIZE_X = 150
SIZE_Y = 150
SIZE_Z = 150
PT_DROP = 0.5    # target z drop per pt
PATH_COUNT = 3 # Number of paths to generate
SCREW_RAD = 12 # Center of rotation to center of marble on track
SCREW_PITCH = 40 # mm per rev
LIFT_SUPPORT_PTS = 21
MIRROR_PATHS = False
RANDOM_CNT = 6  # How many random points to generate if LESS_RANDOM_INIT_PATH

Here's a large 4 track one - I think in the original defs.py from Will M.

SIZE_X = 330
SIZE_Y = 175
SIZE_Z = 300
PT_DROP = 0.85    # target z drop per pt
PATH_COUNT = 4 # Number of paths to generate
SCREW_RAD = 18 # Center of rotation to center of marble on track
SCREW_PITCH = 24 # mm per rev
LIFT_SUPPORT_PTS = 51
MIRROR_PATHS = False
RANDOM_CNT = 15  # How many random points to generate if LESS_RANDOM_INIT_PATH

Here's my large one, I haven't printed it yet - sized for my Bambu printer

# Overall size of box to generate path in
SIZE_X = 250
SIZE_Y = 200
SIZE_Z = 250
PT_DROP = 0.8    # target z drop per pt
PATH_COUNT = 6 # Number of paths to generate
SCREW_RAD = 18 # Center of rotation to center of marble on track
SCREW_PITCH = 24 # mm per rev
LIFT_SUPPORT_PTS = 51
MIRROR_PATHS = False
RANDOM_CNT = 15  # How many random points to generate if LESS_RANDOM_INIT_PATH
