cmake_minimum_required(VERSION 2.8)
project(style_transfer)

set(CMAKE_BUILD_TYPE "Debug")
# set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS_DEBUG "-std=c++11 -g -Wall ${CMAKE_CXX_FLAGS_DEBUG}")
# set(CMAKE_CXX_FLAGS_RELEASE "-std=c++11 ${CMAKE_CXX_FLAGS_RELEASE}")
set(CMAKE_EXE_LINKER_FLAGS "-lpthread -fPIC ${CMAKE_EXE_LINKER_FLAGS}")

find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui)
include_directories(${OPENCV_INCLUDE_DIR})
include_directories($ENV{NEUWARE}/include/)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

#link_directories(${CMAKE_CURRENT_SOURCE_DIR}/lib)
link_directories($ENV{X86_LIB_PATH})
link_directories($ENV{NEUWARE}/lib64/)
link_libraries("libcnrt.so")
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/bin)

add_executable(style_transfer src/style_transfer.cpp 
               src/data_provider.cpp 
               src/inference.cpp 
               src/post_processor.cpp)


target_link_libraries(style_transfer ${OpenCV_LIBS})
