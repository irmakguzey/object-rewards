<h1 align="center" style="font-size: 2.0em; font-weight: bold; margin-bottom: 0; border: none; border-bottom: none;">HuDOR: Bridging the Human-to-Robot Dexterity Gap through Object-Oriented Rewards</h1>

##### <p align="center"> [Irmak Guzey](https://irmakguzey.github.io/), [Yinlong Dai](https://yinlongdai.github.io/), [Georgy Savva](https://georgysavva.github.io/), [Raunaq Bhirangi](https://raunaqbhirangi.github.io/), [Lerrel Pinto](https://lerrelpinto.com)</p>
##### <p align="center"> New York University </p>

#####
<div align="center">
    <a href="https://object-rewards.github.io"><img src="https://img.shields.io/static/v1?label=Project%20Page&message=Website&color=blue"></a> &ensp;
    <a href="https://arxiv.org/abs/2410.23289"><img src="https://img.shields.io/static/v1?label=Paper&message=Arxiv&color=red"></a> &ensp; 
    <a href="https://osf.io/frdc9/"><img src="https://img.shields.io/static/v1?label=Data&message=OSF&color=orange"></a> &ensp;
    
</div>

#####


<!-- <p align="center">
  <img width="45%" src="https://github.com/see-to-touch/see-to-touch.github.io/blob/main/mfiles/gifs/sponge_flipping.gif">
  <img width="45%" src="https://github.com/see-to-touch/see-to-touch.github.io/blob/main/mfiles/gifs/eraser_turning.gif">
 </p>

 <p align="center">
  <img width="45%" src="https://github.com/see-to-touch/see-to-touch.github.io/blob/main/mfiles/gifs/mint_opening.gif">
  <img width="45%" src="https://github.com/see-to-touch/see-to-touch.github.io/blob/main/mfiles/gifs/peg_insertion.gif">
</p> -->

This repository includes the official implementation of [HuDOR](https://object-rewards.github.io). It includes the human-to-robot reward calculation mechanism, online imitation algorithm and offline imitation baselines. Our setup enables human-to-robot policy learning transform for 4 different dexterous tasks showed above. 

Our hardware setup consists of an [Allegro hand](https://www.wonikrobotics.com/research-robot-hand) and a [Kinova arm](https://assistive.kinovarobotics.com/product/jaco-robotic-arm). Demonstrations are collected using a forked version of [Open-Teach](https://open-teach.github.io) pipeline that integrates human fingertips to the environment and rewards are calculated using a mixture of [Lang-SAM](https://github.com/luca-medeiros/lang-segment-anything) and [Co-Tracker](https://co-tracker.github.io). All the other github repositories used are added as a submodules and installations can be reached from below.

---

All data is streamed at an [OSF](https://osf.io/frdc9/) project, you can either download data using the webpage or using the `wget` terminal command as 

```
wget -O desired-file-location> https://osf.oi/<file-id-in-osf>/download
```

## Installation

* `git clone --recurse-submodules https://github.com/irmakguzey/object-rewards.git` to install the package.
* Install the conda environment by:
`conda env create -f environment.yml` 
* Submodules are located at `submodules/`, install each submodule by running `pip install -e .` on their codebase. Also follow instructions on their README if there is anything needed.
* Install `object_rewards` library by: `pip install -e .` 

### Checkpoint Installation

You need to download CoTracker2 checkpoint in order to run reward calculation. You can either download this using the instructions that they provide in [Co-Tracker](https://co-tracker.github.io) or you can download it by using the project link: 

```
wget -O checkpoints/cotracker2.pth https://osf.io/guwf7/download
```

You can also download every file streamed at data link with the command: `wget -O <des-file-name> https://osf.io/<file-id>/download` .

## Testing Object Rewards 

* You can also run example reward calculations using the scripts provided in `examples` folder. 
* In these scripts `get_reward.py` and `get_segmented_video.py` you will see examples of how to segment a video provided with a language prompt and get trajectory rewards. 
* We also provide human and robot videos used in our projects in that same folder. You can run these scripts and observe outputs. 
* Running these scripts will output segmented videos and reward information images such as segmentation masks, and plots of trajectory similarities between given referance video and the evaluation video.

## Reproducing Results 

**Start camera servers:** Before reproducing any of the results you would need to start the camera servers using [Open-Teach](https://open-teach.github.io). Visit their page for more instructions. 

### Calibration

* You need to calibrate the robot to the environment before using this code on your setup. 
* For that, stick an ArUCo marker on top of the hand and save the transform from that marker to the end effector of the arm. This transform is used in `CalibrateBase.get_aruco_corners_in_3d`. You would need to change every point `p1, p2, p3, p4` according to that transform.
* Then, you can use the following code piece for calibration: (this can also be found in `examples/calibrate.py`)
```
from object_rewards.calibration import CalibrateBase

HOST = "172.24.71.240"
CALIBRATION_PICS_DIR = "<calibration-dic>"
CAM_IDX = 0
MARKER_SIZE = 0.05

base_calibr = CalibrateBase(
    host=HOST,
    calibration_pics_dir=CALIBRATION_PICS_DIR,
    cam_idx=CAM_IDX,
)

base_calibr.save_poses()
base_to_camera = base_calibr.calibrate(True, True)
base_calibr.get_calibration_error_in_2d(base_to_camera=base_to_camera)

```
* Method `save_poses` will ask you to move the arm manually to 50 different locations. At every new location you will be asked to press enter. Move the arm to locations where the camera can see the aruco marker. Try to cover all possible arm locations.
* When you press enter, camera will take a picture and save both the aruco pose and the arm pose. 
* Using these corresponding transforms, camera to robot base transform gets calculated using `cv2.solvePnp` method.

### Data Collection
If you'd like to have the HumanBot app, please fill out [this Google form](https://docs.google.com/forms/d/e/1FAIpQLSd8_ZLIhLAyQ4EphTYnUto0lZgtgRqTmxd7ZraQIqAh2eRNkw/viewform?usp=header) and we'll get back to you with instructions to get the app. 
After downloading and installing the app, to collect human demonstrations, you can run:

```bash
python submodules/Open-Teach-HuDOR/data_collect.py storage_path=<desired-storage-path> demo_num=<demo-number>
```

Further adjustments to data collection parameters can be made in the Hydra config file located at: `submodules/Open-Teach-HuDOR/configs/collect_data.yaml`.

During data collection, follow these steps:
* Focus on the ArUco marker on the operation table for approximately 5-6 seconds before starting object manipulation. This helps establish the transformation between the VR headset and the world frame.
* Pinch your right index and thumb fingers to indicate the start of the demonstration.
* Proceed to manipulate the object.
* Pinch your right index and thumb fingers again to signal the end of the demonstration.

### Online Training

* After downloading the collected task demonstrations or collecting your own tasks, you can run `python train_online.py` to start training residual models. You can download tasks and checkpoints using `wget -O <des-file-name> https://osf.io/<file-id>/download`.
* All parameters of the online training module can be modified in `configs/train_online.yaml`.

### Offline Training

* We provide implementations of [VQ-Bet](https://sjlee.cc/vq-bet/) offline baseline using a PointCloud as input. 
* After collecting demonstrations, you can run `python train_offline.py learner=vq_vae` to train codebooks using VAE. This part learns offline *skills* about the task. 
* Then you can train the Bet policies atop these skills using `python train_offline.py learner=vq_bet`. 
* More configs can be modified in `configs/train_offline.yaml` 


