

import igl
import numpy as np

timing_file = "C:\\Users\\otmanbench\\Desktop\\fastCD\\fast_cd_cpp\\results\\full_vs_reduced\\cd_fish\\full\\timings.DMAT"

t = igl.read_dmat(timing_file)

print("timing", t.mean())
print("fps", 1/t.mean())