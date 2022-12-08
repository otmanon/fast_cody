import numpy as np
import igl
import fast_cd_pyb as fcd

name = "elephant"
root_dir = "C:/Users/otmanbench/Desktop/fastCD/fast_cd_cpp/data/raw_data/" + name + "/";
msh_file = root_dir + "/" + name +".msh";

[V, T] = igl.read_msh(msh_file);
fcd.writeDMATd(root_dir + "V.DMAT", V );
fcd.writeDMATi(root_dir + "T.DMAT", T );