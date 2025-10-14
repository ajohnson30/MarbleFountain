# Marble Fountain Generator

This is a repository of code to procedurally generate large complex organic marble runs. 

This codebase is super bodged together and I give no garuntees of y'all being able to anything working. The whole thing desperately needs a massive cleanup and I am way outside the level of complexity that OpenSCAD is the best tool for. That said, I figured folks would prefer that I share it as is than not at all. If I do take the time to refactor I'm likely going to do a substantial rewrite and use a different tool for mesh generation. 

## Commands

Run full generation in one line `python3 newPath.py output/ SOLVE && python3 pathGen.py && python3 modelGen.py && python3 makeSTLs.py`

`python3 newPath.py output/ SOLVE` re-initializes the paths

`python3 pathGen.py` runs the solver

`python3 modelGen.py` generates track and support geometry into scad files

`python3 makeSTLs.py` generates stls from scad files

While `pathGen.py` runs `python3 shared.py` will plot the paths as they are solved

`defs.py` defines a ton of static variables for configuring the generation, some of which have comments

## Caveats

This only runs on Linux (WSL should work)

The screw and base are sized to fit a nema 17 stepper motor glued onto the base

Everything is tuned around 1/4" ball bearings and you may need to retune things for anything else

`makeSTLs.py` assumes you have OpenSCAD at `~/install/OpenSCAD.AppImage`

Some of the random knobs in `defs.py` have not been touched in over a year and I don't remember what they do
