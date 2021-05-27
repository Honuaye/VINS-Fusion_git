cmake_minimum_required(VERSION 2.8.3)
project(vins)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++11")
#-DEIGEN_USE_MKL_ALL")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")
# Make sure that custom modules like FindXXXX are found
# camera_models
find_package(catkin REQUIRED COMPONENTS
    roscpp
    std_msgs
    geometry_msgs
    nav_msgs
    tf
    cv_bridge
    camera_models
    image_transport)
find_package(GTSAM REQUIRED)
find_package(OpenCV REQUIRED)
# find_package(Eigen3)
# message(WARNING "OpenCV_VERSION: ${OpenCV_VERSION}")

include_directories(
    ${catkin_INCLUDE_DIRS}
)

# set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
# find_package(Eigen3)
include_directories(
  ${catkin_INCLUDE_DIRS}
#   ${EIGEN3_INCLUDE_DIR}
  ${GTSAM_INCLUDE_DIR}
)

catkin_package()

add_library(vins_lib
    src/estimator/parameters.cpp
    src/estimator/stereo_camera.cpp
    src/estimator/imu_frontend.cpp
    src/estimator/estimator.cpp
    src/estimator/feature_manager.cpp
    src/utility/utility.cpp
    src/utility/visualization.cpp
    src/utility/CameraPoseVisualization.cpp
    src/initial/solve_5pts.cpp
    src/initial/initial_aligment.cpp
    src/initial/initial_ex_rotation.cpp
    src/featureTracker/feature_tracker.cpp
)

set(GTSAM_LIBRARIES gtsam)   # TODO: automatic search libs
find_package(GTSAMCMakeTools)
include(GtsamMakeConfigFile)
include(GtsamBuildTypes)

target_link_libraries(
    vins_lib ${catkin_LIBRARIES} ${OpenCV_LIBS}
    gtsam
    gtsam_unstable
)

add_executable(vins_node src/rosNodeTest.cpp)
target_link_libraries(vins_node vins_lib) 

#add_executable(kitti_odom_test src/KITTIOdomTest.cpp)
#target_link_libraries(kitti_odom_test vins_lib) 
#add_executable(kitti_gps_test src/KITTIGPSTest.cpp)
#target_link_libraries(kitti_gps_test vins_lib) 

