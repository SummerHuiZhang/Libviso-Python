# project
cmake_minimum_required (VERSION 2.8)
project (libviso2)
# directories
set (LIBVISO2_SRC_DIR src)
set (PYBIND_DIR pybind11)
set(CMAKE_CXX_STANDARD 11)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/pybind11/tools")

set(DPYBIND11_PYTHON_VERSION 3.7.7)
set(Python_ADDITIONAL_VERSIONS 3.9 3.8 3.7 3.6 3.5 3.4)
find_package(PythonLibsNew ${PYBIND11_PYTHON_VERSION} REQUIRED)
# include directory
include_directories(
"${LIBVISO2_SRC_DIR}"
${PROJECT_SOURCE_DIR}/pybind11/include
/usr/local/Cellar/png++/0.2.10/include
/usr/local/include/
${PYTHON_INCLUDE_DIRS}
)

link_directories(
/usr/local/lib/
)

# use sse3 instruction set
SET(CMAKE_CXX_FLAGS "-msse3")

# sources
FILE(GLOB LIBVISO2_SRC_FILES "src/*.cpp")

# make release version
set(CMAKE_BUILD_TYPE Release)
# demo program
#add_executable(viso2 ${LIBVISO2_SRC_FILES})
#target_link_libraries (viso2 png)

#add_subdirectory(pybind11)
find_package(pybind11 REQUIRED)
pybind11_add_module(pyviso
src/pyviso.cpp
src/viso.cpp
src/viso_mono.cpp
src/filter.cpp
src/matcher.cpp
src/matrix.cpp
src/triangle.cpp
)
target_link_libraries(pyviso
png
${PYTHON_LIBRARIES}
    )

