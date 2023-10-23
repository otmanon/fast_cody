# Fast Complementary Dynamics Code
We're currently working on a pip installation, but for now, this is how you can get started with our codebase.

Clone this repo and all its submodules:
```
git clone --recursive https://github.com/otmanon/fast_cd_pyb
```

Inside the repository above, install the dependencies:
```
pip install -r requirements.txt
```

Finally, build the library from source by running:
```
python setup.py install
```


# Apps
We provide a variety of fast_cd apps shown in our paper.

## Interactive Affine Handle
Run 
```
import fast_cd_pyb as fcd
fcd.apps.interactive_cd_affine_handle()
```

This should run a few computations, and then finally open a window with the 
classic Complementary Dynamics fish. By playing with the Guizmo, you can interact with 
the fish. Press `g` to change guizmo transorm operations.

We also provide a few different meshes. The demo found in `demos/affine_handle.py` shows how to change some of the inputs we 
give to the `interactive_cd_affine_handle` such as changing the subspace size, reading from cache,
or using different meshes.

We also provide many example meshes in the `data` directory. To run the demo on a `.msh` file of your choice, run

```
fcd.apps.interactive_cd_affine_handle(msh_file_path)
```
