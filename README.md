# Fast Complementary Dynamics Python Bindings!
Clone this repo and all its submodules:
```
git clone --recursive https://github.com/otmanon/fast_cd_pyb
```

# Python Environment

This repo has some python bindings for our fast CD project! When working with these bindings I recommend using the [Anaconda](https://www.anaconda.com/) Virtual Environment manager for python. Once you've install anaconda, open a terminal and create a virtual environment.

```
conda create -n <ENV_NAME> python=3.8
```

Then,  
```
conda activate <ENV_NAME>
```

I'm currently using python3.8 but any version should work.

## Dependencies
Before we compile the source code from scratch, let's do the easy step
```
conda install numpy 
conda install scipy
conda install -c conda-forge igl
pip install mediapipe

```

## Getting a Python Module of fast_cd in our Environment
Of course, the main dependency is our [fast_cd](https://github.com/otmanon/fast_cd.git)  C++ codebase, which exposes functions useful for the implementation of Fast Complementary Dynamics, as well as wrapper code for the libigl viewer.

We use [pybind11](https://github.com/pybind/pybind11/) for this binding code.  

It should be as easy as running :
```
python setup.py install
``` 
This will compile fast_cd and it's dependencies, like libigl. 

That should be it, you should now be able to check that you've installed the python bindings by running

```
python
import fast_cd_pyb as fcd
viewer = fcd.fast_cd_viewer()

viewer.launch()
```

## Demos
There are three demos you can run out of the box(Give or take changing the directory of the source files you give it in the `data` folder.

```
python tests/interactive_cd_affine_handle.py
```

Exposes control of the affine handle of a  shape using the Imgui Guizmo

```
python tests/cd_demo_face_tracking.py
```

Opens your laptop's camera using mediapipe, fits a rotation to your face mesh and uses that rotation as rig motion.

```
python tests/cd_demo_pose_tracker.py
```

Lets you control charizard's wings with rotations fitted to mediapipes arm landmarks. When you launch it, be sure to already be standing in front of the camera with a T-pose, and that both your arms are clearly visible to the camera. 




## Making Changes to the Bindings
All the actual "binding" code is written in `src/core.cpp`

You can add whatever functions you like to this file, a good idea is to consult prior functions/classes, and also to read the pybind11 startup manual.

If you make changes to the bindings (Like writing new bounded functions), the same command should run much faster. 
Either that or run 
```
python setup.py develop
```

If you want to clean the cache and rebind it from scratch the long way, call
```
python setup.py clean
```

