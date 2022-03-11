cmake_minimum_required(VERSION 3.0.2)
project(adversary_cbf)

add_compile_options(-std=c++11)

find_package(catkin REQUIRED COMPONENTS
	roscpp
  	std_msgs
	nav_msgs
	geometry_msgs
	tf2
	)

find_package(Eigen3)

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES tracker
#  CATKIN_DEPENDS other_catkin_pkg
#  DEPENDS system_lib
)

include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)

#add_dependencies(${PROJECT_NAME} ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

add_executable(${PROJECT_NAME}_node src/tracker_node.cpp)
add_dependencies(${PROJECT_NAME}_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

target_link_libraries(${PROJECT_NAME}_node
   ${catkin_LIBRARIES}
   OsqpEigen::OsqpEigen
 )

 add_executable(flat_space_tracker_node src/flat_space_tracker_node.cpp)

add_dependencies(flat_space_tracker_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

target_link_libraries(flat_space_tracker_node
   ${catkin_LIBRARIES}
   OsqpEigen::OsqpEigen
 )


add_executable(trajectoryPublisher src/DItrajectoryPublisher.cpp)
add_dependencies(trajectoryPublisher ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(trajectoryPublisher  ${catkin_LIBRARIES} )


add_executable(ekf src/ekf.cpp)
add_dependencies(ekf ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS}) 
target_link_libraries(ekf ${catkin_LIBRARIES} OsqpEigen::OsqpEigen)

add_executable(vicon_velocity_node src/vicon_velocity.cpp)
add_dependencies(vicon_velocity_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS}) 
target_link_libraries(vicon_velocity_node ${catkin_LIBRARIES})
