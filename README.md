# Landmark Localization

## Dependencies
* Ubuntu 20.04
* ROS Noetic

## Python Packages
* [NumPy](https://numpy.org/)
* [SciPy](https://scipy.org/)
* [PyYAML](https://pypi.org/project/PyYAML/)
* [Matplotlib](https://pypi.org/project/matplotlib/)


## Setup
1. Open a terminal, run ```roscore```.
2. Check `config/settings.yaml`, ensure the `filter_name` is set to `test`.
3. Open a new terminal, run ```rviz```.
4. Open a visualization config file. In your rviz, click `file` -> `open config`, choose `rviz/default.rviz` in the homework folder.
5. Open a new terminal, run ```python3 run.py```.
6. You should be able to see your a robot moving in `rviz`.

PS. A dummy filter is provided to test if environment is set up correctly.
<!-- 
**Note:** We include a dummy filter in the code, which allows you to test if you have set up your environment correctly. To run the dummy filter, set `filter_name` to `test` in `config/settings.yaml` and do `python3 run.py`. -->


## Configurations
Parameters can be modified in `config/settings.yaml`.
* `filter_name`: The filter you would like to run. Options include: `EKF`,`UKF`, `PF`, `InEKF`, and `test`.


## Results
<img src="img/result-ekf.gif" width="100%" height="100%"/>

* `Green path` represents command path without action noise
* `Blue path` represents the actual path that the robot moves due to action noise
* `Red arrow` represents the estimated robot pose
* `Red ellipse` represents the covariance


## Visualization
1. open rviz in a terminal
2. In rviz, click `file` -> `open config`.
3. Choose `rviz/default.rviz`.
4. Open a new terminal and run your filter.
