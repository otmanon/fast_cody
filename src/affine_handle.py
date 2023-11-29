from fast_cody.apps import *
from fast_cody import get_data
# name = "./data/dromedary/dromedary.msh"
# interactive_cd_affine_handle(name, read_cache=False, num_modes=16)


# name = "./data/xyz_dragon/xyz_dragon.msh"
# interactive_cd_affine_handle(name,  read_cache=False, num_modes=16)

# name = "./data/elephant/elephant.msh"
# interactive_cd_affine_handle(name, rho=1000, read_cache=False, num_modes=16)


# interactive_cd_affine_handle()


# name = "../data/bulldog/bulldog.msh"
# texture_obj = "../data/bulldog/bulldog_tex.obj"
# texture_png = "../data/bulldog/bulldog_tex.png"
# cache_dir = "../data/bulldog/cache/"
# interactive_cd_affine_handle(name, mu = 3e4, rho=1000, read_cache=False, num_modes=16,
#                              texture_obj=texture_obj, texture_png=texture_png,
#                              cache_dir=cache_dir)


name = "../data/king_ghidora/king_ghidora.msh"
texture_obj = "../data/king_ghidora/king_ghidora_tex.obj"
texture_png = "../data/king_ghidora/king_ghidora_tex.png"
cache_dir =  "../data/king_ghidora/cache/"
interactive_cd_affine_handle(name, texture_png=texture_png,
                             texture_obj=texture_obj,
                             read_cache=False, cache_dir=cache_dir)



