import igl

#unconstrained vs constrained
file = "C:/Users/otmanbench/Desktop/fastCD/fast_cd_cpp/results/constrained_vs_unconstrained/elephant_high_res/unconstrained_modes_constrained_sim/kinetic_energy_complementary.DMAT"

#constrained vs unconstrained
# file = "C:/Users/otmanbench/Desktop/fastCD/fast_cd_cpp/results/constrained_vs_unconstrained/elephant_high_res/constrained_modes_unconstrained_sim/kinetic_energy_complementary.DMAT"

v = igl.read_dmat(file);

print(v.mean())