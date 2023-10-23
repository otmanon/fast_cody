cmake_minimum_required(VERSION 3.1)
project(fast_cd_pyb
  DESCRIPTION
  "Fast CD Python Bindings"
)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_BUILD_TYPE "Release")
set(EXT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/deps")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

#get fast_cd library... should really put this in deps and separate the official python bindings from this repo

add_subdirectory(./deps/fast_cd)
add_subdirectory(./deps/pybind11/)

pybind11_add_module(fast_cd_pyb
  "${CMAKE_CURRENT_SOURCE_DIR}/src/cpp/core.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/cpp/viewer.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/cpp/igl.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/cpp/sim.cpp"
)



target_include_directories(fast_cd_pyb PUBLIC ./deps/fast_cd/src/include/)
target_include_directories(fast_cd_pyb PUBLIC ./src/include/)
target_include_directories(fast_cd_pyb PUBLIC ./src/cpp/)

target_link_libraries(fast_cd_pyb PUBLIC fast_cd)

set_target_properties(fast_cd_pyb PROPERTIES LINKER_LANGUAGE CXX)
set_property(TARGET fast_cd_pyb PROPERTY CXX_STANDARD 17)