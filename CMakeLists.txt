cmake_minimum_required(VERSION 3.0)
project(mono_slam)

# Library OpenCV
find_package(OpenCV REQUIRED)

# Library Eigen3
find_package (Eigen3 3.3 REQUIRED NO_MODULE)

set(INCLUDE_DIRECTORY ${PROJECT_SOURCE_DIR}/include)
set(SOURCE_DIRECTORY ${PROJECT_SOURCE_DIR}/include)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${INCLUDE_DIRECTORY})
include_directories(${SOURCE_DIRECTORY})

file(GLOB mono_files
	"include/*.h"
	"src/*.cpp"
)

add_executable(mono_slam ${mono_files})
target_link_libraries(mono_slam ${OpenCV_LIBS} Eigen3::Eigen)
