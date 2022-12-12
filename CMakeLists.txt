cmake_minimum_required(VERSION 3.1)
project(fast_cd_pyb)
#get fast_cd library... shoudl really put this in deps and separate the official python bindings from this repo
add_subdirectory(./deps/fast_cd)
add_subdirectory(./deps/pybind11/)

pybind11_add_module(fast_cd_pyb
  src/core.cpp
  src/viewer.cpp
)
target_include_directories(fast_cd_pyb PRIVATE ./deps/fast_cd/include/)
target_link_libraries(fast_cd_pyb PRIVATE fast_cd)

set_target_properties(fast_cd_pyb PROPERTIES LINKER_LANGUAGE CXX)
set_property(TARGET fast_cd_pyb PROPERTY CXX_STANDARD 17)
