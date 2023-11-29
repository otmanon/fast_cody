# from fast_cd.apps import interactive_cd_face_tracking
import fast_cody.apps
import fast_cody as fcd

import fast_cd_pyb as fcdp

name = "../data/elephant/elephant.msh"
[V, F, T] = fcd.read_msh(name)
fcd.apps.interactive_cd_face_tracking(V=V, T=T)

# fast_cody.apps.interactive_cd_face_tracking()