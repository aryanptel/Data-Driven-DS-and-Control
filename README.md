Generalized Synchronization and Swarming Algorithm Implementation

Overview:  
This repository contains all codes and theoretical calculation I have performed for my MS thesis project on "Design of autonomous swarms". The repository contains code for generalized synchronization analysis of Rössler systems using Mathematica, along with its numerical implementation in Julia. Additionally, it includes the development of a swarming algorithm with a master-slave configuration. 

Generalized Synchronization:  
We explore the concept of generalized synchronization and its application in the development of a swarming algorithm.

Swarming Algorithm:  
We have developed a swarming algorithm implemented in two different systems:

(i) Non-holonomic Robots (Differential-Drive):  
For non-holonomic systems, we utilize a master-slave configuration with two differential-drive robots. Simulation and testing are conducted in Gazebo using ROS (Robot Operating System). The UGV folder contains relevant files and Python scripts for ROS implementation.

(ii) Holonomic Systems (Drones):  
To test the algorithm in holonomic systems, we employ a swarm of drones. The implementation utilizes ArduPilot and Mavproxy platforms. The algorithm is scripted in Python using the pymavlink module. The Drone folder contains a series of Python scripts along with instructions for their execution in chronological order. The implementation has been done on five drones with one as master and other four as slaves.

Coding Language: Julia (numerical solutions), Python (robotics)  

**Please find a video of the implmentation of this algorithm on real drones in UGV folder.**

Instructions:  
Please refer to individual folders for detailed usage instructions and implementations. A detailed overview will be added soon. 

Contributors:
Aryan Patel

License
This project is licensed under the License - see the LICENSE file for details.

Feel free to report issues!
