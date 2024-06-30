cmake_minimum_required (VERSION 3.28.0)
project(yolov10_ncnn LANGUAGES C CXX)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_STANDARD 14)

find_package(OpenCV REQUIRED)
find_package(OpenMP REQUIRED)

if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP_CXX_FOUND Found")
else()
    message(FATAL_ERROR "OpenMP_CXX_FOUND Not Found!")
endif()

if (OpenCV_FOUND)
    message(STATUS "OpenCV_LIBS: ${OpenCV_LIBS}")
    message(STATUS "OpenCV_INCLUDE_DIRS: ${OpenCV_INCLUDE_DIRS}")
else ()
    message(FATAL_ERROR "opencv Not Found!")
endif()

link_directories(${CMAKE_CURRENT_SOURCE_DIR}/lib)

FILE(GLOB SRC_LST src/*.cpp)

add_executable(yolov10_ncnn ${SRC_LST})
target_link_libraries(yolov10_ncnn  
    ${OpenCV_LIBS}
    ncnn
    OpenMP::OpenMP_CXX
    )
target_include_directories(yolov10_ncnn PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}/include/ncnn
    ${CMAKE_CURRENT_SOURCE_DIR}/include/)

