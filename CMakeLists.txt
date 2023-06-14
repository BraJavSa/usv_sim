cmake_minimum_required(VERSION 3.0.2)
project(usv_sim)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  vrx_gazebo
  wamv_gazebo
  xacro
)
catkin_package(
  CATKIN_DEPENDS vrx_gazebo wamv_gazebo xacro
)
# Generate world files from xacro and install
xacro_add_files(
  worlds/dock.world.xacro
  worlds/example_course.world.xacro
  worlds/navigation_task.world.xacro
  worlds/ocean.world.xacro
  worlds/perception_task.world.xacro
  worlds/sandisland.world.xacro
  worlds/scan_and_dock.world.xacro
  worlds/stationkeeping_task.world.xacro
  worlds/wayfinding_task.world.xacro
  worlds/wind_test.world.xacro
  ${XACRO_INORDER} INSTALL DESTINATION worlds
)

# Install all the config files
install(DIRECTORY config/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/config)

# Install all the world files
install(DIRECTORY worlds/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/worlds)

# Install all the launch files
install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch)
include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)


