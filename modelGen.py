#! /usr/bin/env python

from solid2.extensions.bosl2 import circle, cuboid, sphere, cylinder, \
									heightfield, diff, tag, attach, \
									TOP, BOTTOM, CTR, metric_screws, rect, glued_circles, \
									chain_hull, conv_hull, hull, cube, union, trapezoid, teardrop, skin, sweep
from solid2.core import linear_extrude

import numpy as np
from copy import deepcopy
import pickle as pkl

from matplotlib import pyplot as plt
import sys
import os

from defs import *
from shared import *
from openScadGenerators import *


fullPaths = pkl.load(open(WORKING_DIR+'/path.pkl', 'rb'))

# outputAssembly = sphere(0)

# testProfile = sphere(MARBLE_RAD*0.6)
# for fooPath in fullPaths:
#     outputAssembly += getShapePathSet(fooPath, np.zeros_like(fooPath), testProfile)

# outputAssembly.save_as_scad(WORKING_DIR + "out.scad")

outputAssembly = getCenterScrewRotatingPart()
outputAssembly.save_as_scad(WORKING_DIR + "out.scad")