cmake_minimum_required(VERSION 2.8.3)
project(kangaroo_driver)

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  roscpp
  std_msgs
  trajectory_msgs
  tf
)


catkin_package(INCLUDE_DIRS include)

include_directories(include
  ${catkin_INCLUDE_DIRS}
)

## Declare a cpp library
add_library(kang_lib
  src/driver_files/kang_lib.cpp
)

## Declare a cpp executable
add_executable(kangaroo_driver_node src/driver_files/kangaroo_driver_node.cpp src/driver_files/kangaroo_driver.cpp)

## Specify libraries to link a library or executable target against
target_link_libraries(kangaroo_driver_node kang_lib
  ${catkin_LIBRARIES}
)


## Mark executables and/or libraries for installation
install(TARGETS kangaroo_driver_node kang_lib
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

