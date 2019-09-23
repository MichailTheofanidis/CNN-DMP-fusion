# Learning Visuomotor Policies by Combining Movement Primitives and Convolutional Neural Networks

Code to learn end-to-end visuomotor policies for robotic arms from demonstrations. The method computes state-action mappings in a supervised learning manner from raw images and motor commands. At the core of the system, a Convolutional Neural Network (CNN) extracts image features and produces motion features. The motion features encode and reproduce motor commands according to the Dynamic Movement Primitives (DMP) framework.

The data and results can be seen in:
https://github.com/MichailTheofanidis/CNN-DMP-fusion-Datasets-Results

Presentation of the project can be seen in:
https://drive.google.com/file/d/1RcRpl1DFY9KgnQEV72diXTBERXydCJcP/view?usp=sharing


The repository does not contain the ROS packages for the ROS-Unreal communication or the Unreal environment where the experiments took place.


