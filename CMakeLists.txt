
cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

#cmake_policy(SET CMP0077 OLD)

set(PROJECT_NAME "NanoBench")
project(${PROJECT_NAME} LANGUAGES CXX CUDA)  

#default build type is Release
if (CMAKE_BUILD_TYPE STREQUAL "")
    set(CMAKE_BUILD_TYPE Release)
endif ()

SET(CUDA_SEPARABLE_COMPILATION ON)

########## Libraries/flags Starts Here ######################
find_package(CUDA REQUIRED)

include("cmake/AutoDetectCudaArch.cmake")

include_directories("${CUDA_INCLUDE_DIRS}")
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -lineinfo; -std=c++11; -expt-extended-lambda; -O3; -use_fast_math; -rdc=true)
#set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -Xptxas -warn-spills -res-usage)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -Xptxas -v;)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -m64 -Wall -std=c++11")

if(MSVC)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_SCL_SECURE_NO_WARNINGS")
endif()
########## Libraries/flags Ends Here ######################


########## inc/libs/exe/features Starts Here ######################
set(CMAKE_INCLUDE_CURRENT_DIR ON)


file(GLOB_RECURSE MY_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/*.*)
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${MY_SRC})

if (MSVC)
	add_executable(${PROJECT_NAME} "${MY_SRC}")
else()
	CUDA_ADD_EXECUTABLE(${PROJECT_NAME} "${MY_SRC}")
endif()

##add features to the library -> require it to compile with cuda c++11
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_11)

##enable position-indep code 
set_target_properties(${PROJECT_NAME} PROPERTIES POSITION_INDEPENDENT_CODE  ON)

##enable the separate compilation on both device and host sides 
set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(${PROJECT_NAME} ${CUDA_LIBRARIES} cuda cublas)
########## inc/libs/exe/features Ends Here ######################