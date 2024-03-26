# usv_sim

This a gazebo simulation of a USV based of VRX competence. Follow this tutorial for use.

All the proyect was developed on Ubuntu 20.04

## Install ROS Noetic:
  
  ```
  sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
  sudo apt install curl 
  curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add -`
  sudo apt update
  sudo apt install ros-noetic-desktop-full
  echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc
  sudo apt install python3-rosdep python3-rosinstall python3-rosinstall-generator python3-wstool build-essential
  sudo rosdep init
  rosdep update
``` 
then close all terminals and open one again 


## Install MAVROS
  
  ```
  sudo apt-get install ros-noetic-mavros ros-noetic-mavros-extras`
  wget https://raw.githubusercontent.com/mavlink/mavros/master/mavros/scripts/install_geographiclib_datasets.sh
  sudo bash ./install_geographiclib_datasets.sh 
  
  ```

## Install SIMULATOR

    ```
    sudo apt-get install ros-noetic-gazebo-ros-pkgs ros-noetic-gazebo-ros-control
    sudo apt-get install ros-noetic-hector-gazebo-plugins
    http://wiki.ros.org/joy/Tutorials/ConfiguringALinuxJoystick
    sudo apt-get install -y libgazebo11-dev
    mkdir -p ~/catkin_ws/src
    cd ~/catkin_ws/src
    catkin_init_workspace
    cd ~/catkin_ws
    catkin_make
    echo "source ~/catkin_ws/devel/setup.bash" >> ~/.bashrc
    cd ~/catkin_ws/src
    git clone https://github.com/BraJavSa/hector-quadrotor-noetic.git
    git clone https://github.com/BraJavSa/px4_offboard_control.git
    git clone https://github.com/BraJavSa/usv_sim.git
    git clone -b gazebo_classic https://github.com/BraJavSa/vrx.git 
    catkin_make
    ```
## run SIMULATOR

for manual mode:

    ```
    roslaunch usv_sim manual_sandisland.launch
    ```

for multi robot mode:

    ```
    roslaunch usv_sim hector_usv_sandisland.launch
    ```

For position control based on polar coordinates:

    first terminal
    
    ```
    roslaunch usv_sim control_position.launch
    ```
    
    open a second terminal and use the "/boat/posed" topic for set a new position

if you want to create a new control script you can use:

    ```
    roslaunch usv_sim sandisland.launch
    ```
Look at "rosrun pxe_offboard_control prueba.py" as an example