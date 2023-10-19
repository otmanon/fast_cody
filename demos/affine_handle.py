

from fast_cd.examples import *




name = "./data/dromedary/dromedary.msh"
interactive_cd_affine_handle(name, mu = 10, read_cache=False, num_modes=16)




name = "./data/xyz_dragon/xyz_dragon.msh"
interactive_cd_affine_handle(name, mu = 10, read_cache=True, num_modes=16)



name = "./data/elephant/elephant.msh"
interactive_cd_affine_handle(name, mu = 10, read_cache=True, num_modes=16)


name = "./data/bulldog/bulldog.msh"
interactive_cd_affine_handle(name, mu = 10, read_cache=True, num_modes=16)


name = "./data/cd_fish/cd_fish.msh"
interactive_cd_affine_handle(name,  mu=100, read_cache=True, num_modes=16)