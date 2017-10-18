cmake_minimum_required(VERSION 2.8)

project(Project2)

SET (CMAKE_CXX_FLAGS "-std=c++11 -D_GLIBCXX_USE_CXX11_ABI=0")
SET (EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_BINARY_DIR}/bin)

set (third_party_install_path ${CMAKE_CURRENT_BINARY_DIR}/lapack)

include(ExternalProject)
ExternalProject_Add(lapack3
	SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/lapack-3.7.1-src/
	CMAKE_COMMAND cmake
	CMAKE_ARGS -DLAPACKE=ON -DCBLAS=ON -DBUILD_TESTING=OFF -DBUILD_SHARED_LIBS=OFF -DCMAKE_INSTALL_PREFIX=${third_party_install_path}
	BUILD_ALWAYS 0
	)

include_directories(src ${CMAKE_CURRENT_BINARY_DIR}/lapack/include)
link_directories(${CMAKE_CURRENT_BINARY_DIR}/lapack/lib64)
file(GLOB_RECURSE source_files ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
add_executable(project2 ${source_files})
target_link_libraries(project2 lapacke lapack cblas blas gfortran)