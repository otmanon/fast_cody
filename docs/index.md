# Fast Complementary Dynamics

This is the documentation page for the source code for [Fast Complementary Dynamics
via Skinning Eigenmodes](https://www.dgp.toronto.edu/projects/fast_complementary_dynamics_site/).

This documentation and codebase are still a work in progress, please report any bugs or features to the [github issues page](https://github.com/otmanon/fast_cody/issues).

# Installation

Create a fresh conda environment with python >= 3.8

```
conda create -n fast_cody python=3.8
conda activate fast_cody
pip install fast-cody
```

# Demo Applications
We provide a few out of the box demos.

###  [Interactive Face Tracking](./apps/interactive_cd_face_tracking.md)


Using mediapipe, you can control a digital avatar with your face, while observing secondary effect!

```
import fast_cody as fcd
fcd.apps.interactive_cd_face_tracking()
```

![Interactive Face Tracking](./imgs/fish_demo_face_tracking.gif)

###  [Interactive Affine Handle](./apps/interactive_cd_affine_handle.md)
Move a single character by manipulating an affine handle
```
import fast_cody as fcd
fcd.apps.interactive_cd_affine_handle()
```

![Interactive Face Tracking](./imgs/fish_demo_affine_handle.gif)

###  [Augmenting Rig Animation](./apps/interactive_cd_rig_anim.md)
Display a rig animation augmented with secondary effects!
```
import fast_cody as fcd
fcd.apps.interactive_cd_rig_anim()
```

![Interactive Face Tracking](./imgs/fish_demo_rig_anim.gif)

