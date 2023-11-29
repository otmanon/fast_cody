# from fast_cody.apps.interactive_cd_rig_anim import interactive_cd_rig_anim
import fast_cody as fcd

## Test input msh_file
msh_file = fcd.get_data("./cd_fish.msh")
# fcd.apps.interactive_cd_rig_anim(msh_file)


# Test input V and T
[V, F, T] = fcd.read_msh(msh_file)
# fcd.apps.interactive_cd_rig_anim(V=V, T=T)


# Test input Wp and P0
import igl
import numpy as np
rig_file = fcd.get_data("./cd_fish_rig.json")
[Vpsurf, Fpsurf, Wpsurface, P0, lengths, pI] = fcd.read_rig_from_json(rig_file)
aI = np.arange(V.shape[0])
[D2, bI, CP] = igl.point_mesh_squared_distance(Vpsurf, V, aI)
Wp = fcd.diffuse_weights(V, T, Wpsurface, bI, dt=10000)

[V, F, T] = fcd.read_msh(msh_file)
# fcd.apps.interactive_cd_rig_anim(V=V, T=T, Wp=Wp, P0=P0)


# Test input P
anim_file = fcd.get_data("./cd_fish_rig_anim__swim.json")
P = fcd.read_rig_anim_from_json(anim_file)
fcd.apps.interactive_cd_rig_anim(V=V, T=T, Wp=Wp, P0=2*P0, P=2*P)