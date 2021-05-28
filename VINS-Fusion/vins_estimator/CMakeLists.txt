cmake_minimum_required(VERSION 2.8.3)
project(vins)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++11")
#-DEIGEN_USE_MKL_ALL")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")
# Make sure that custom modules like FindXXXX are found
# # Prefer to use system installation of gflags/glog
set(GFLAGS_PREFER_EXPORTED_GFLAGS_CMAKE_CONFIGURATION TRUE)
# set(GLOG_PREFER_EXPORTED_GLOG_CMAKE_CONFIGURATION TRUE)
find_package(catkin REQUIRED COMPONENTS
    roscpp
    std_msgs
    geometry_msgs
    nav_msgs
    tf
    cv_bridge
    camera_models
    image_transport)
find_package(Gflags REQUIRED)
# find_package(Glog 0.3.5 REQUIRED)

# Find GTSAM components
find_package(GTSAM REQUIRED) # Uses installed package


find_package(OpenCV REQUIRED)

# message(WARNING "OpenCV_VERSION: ${OpenCV_VERSION}")

find_package(Ceres REQUIRED)

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${CERES_INCLUDE_DIRS}
  ${GTSAM_INCLUDE_DIR}
)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
find_package(Eigen3)
include_directories(
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}

  #   ${GLOG_INCLUDE_DIRS}
  ${GFLAGS_INCLUDE_DIRS}
)

catkin_package()


set(GTSAM_LIBRARIES gtsam)   # TODO: automatic search libs
find_package(GTSAMCMakeTools)
include(GtsamMakeConfigFile)
include(GtsamBuildTypes)
# Boost - same requirement as gtsam
find_package(Boost 1.50 REQUIRED)
include_directories(${Boost_INCLUDE_DIR})

add_library(vins_lib
    src/estimator/parameters.cpp
    src/estimator/estimator.cpp
    src/estimator/feature_manager.cpp
    src/factor/pose_local_parameterization.cpp
    src/factor/projectionTwoFrameOneCamFactor.cpp
    src/factor/projectionTwoFrameTwoCamFactor.cpp
    src/factor/projectionOneFrameTwoCamFactor.cpp
    src/factor/marginalization_factor.cpp
    src/utility/utility.cpp
    src/utility/visualization.cpp
    src/utility/CameraPoseVisualization.cpp
    src/initial/solve_5pts.cpp
    src/initial/initial_aligment.cpp
    src/initial/initial_sfm.cpp
    src/initial/initial_ex_rotation.cpp
    src/featureTracker/feature_tracker.cpp)

target_link_libraries(
    vins_lib ${catkin_LIBRARIES} ${OpenCV_LIBS}
    ${CERES_LIBRARIES}
    # ${GLOG_LIBRARIES}
    ${GFLAGS_LIBRARIES}
    ${Boost_LIBRARIES}
)


add_executable(vins_node src/rosNodeTest.cpp)
target_link_libraries(vins_node vins_lib) 

#add_executable(kitti_odom_test src/KITTIOdomTest.cpp)
#target_link_libraries(kitti_odom_test vins_lib) 

#add_executable(kitti_gps_test src/KITTIGPSTest.cpp)
#target_link_libraries(kitti_gps_test vins_lib) 

