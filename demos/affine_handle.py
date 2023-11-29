from fast_cody.apps import interactive_cd_affine_handle
from fast_cody import get_data
import fast_cd_pyb as fcdp
import fast_cody as fcd


# name = "./data/dromedary/dromedary.msh"
# interactive_cd_affine_handle(name, read_cache=False, num_modes=16)


# name = "./data/xyz_dragon/xyz_dragon.msh"
# interactive_cd_affine_handle(name,  read_cache=False, num_modes=16)

# name = "./data/elephant/elephant.msh"
# interactive_cd_affine_handle(name, rho=1000, read_cache=False, num_modes=16)

name = "./data/elephant/elephant.msh"
[V, F, T] = fcd.read_msh(name)
interactive_cd_affine_handle(V=V, T=T)
#
# name = get_data("./king_ghidora/king_ghidora.msh")
# texture_obj = get_data("./king_ghidora/king_ghidora_tex.obj")
# texture_png = get_data("./king_ghidora/king_ghidora_tex.png")
# interactive_cd_affine_handle(name, texture_png=texture_png,
#                              texture_obj=texture_obj, read_cache=True)
#
#
# name = get_data("./bulldog/bulldog.msh")
# texture_obj = get_data("./bulldog/bulldog_tex.obj")
# texture_png = get_data("./bulldog/bulldog_tex.png")
# interactive_cd_affine_handle(name, mu = 5e3, rho=1000, read_cache=False, num_modes=16,
#                              texture_obj=texture_obj, texture_png=texture_png,
#                              constraint_enforcement="optimal")
#
# #
# name = get_data("./cd_fish/cd_fish.msh")
# interactive_cd_affine_handle(name, read_cache=False, num_modes=16, constraint_enforcement="optimal")