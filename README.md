# RARA: Zero-shot Sim2Real Visual Navigation with Following Foreground Cues

FGBG (foreground-background) pytorch package for defining and training models.
For a demo, please watch: https://youtu.be/nnnhLXBl8J8


## Install Imitation-learning codebase for data collection and evaluation in simulation
See instruction here: https://github.com/kkelchte/imitation-learning-codebase.
If the installation went fluently you should be able to create a dataset from within your sourced singularity environment:
```shell
python3.8 src/sim/ros/src/data_collection_fg_bg.py
```
This will create a json and hdf5 file of a number of flewn trajectories in the line world.


## Install FGBG in a conda environment

```bash
conda create --yes --name venv python=3.6
conda activate venv
conda install --yes --file requirements-conda
conda install --yes pytorch torchvision cudatoolkit=11.0 -c pytorch 
python -m pip install -r requirements-pip
```

## Train your models for extracting the foreground and background

Pretrain a model with bg augmentation from MITplaces stored in data/datasets/places
```bash
python run.py --config_file configs/deep_supervision_triplet.json --texture_directory data/datasets/places --target line --output_dir data/mymodel
```

Finetune the final layers for waypoint prediction with
```bash
python run.py --config_file configs/deep_supervision_triplet.json --texture_directory data/datasets/places --target line --encoder_ckpt_dir data/mymodel --output_dir data/mymodel/waypoints --task waypoints
```

## Evaluate neural network on both simulated and real bebop drone

From within the singularity environment, you can run the following files.
Make sure you adjust each file to the correct task (waypoints) and the correct checkpoint directory (data/mymodel/waypoints).

For evaluation in simulation:
```bash
python3.8 src/sim/ros/src/online_evaluation_fgbg.py
```
For evaluation on the real bebop drone, make sure you connect to the wifi of the drone before launching:
```bash
python3.8 src/sim/ros/src/online_evaluation_fgbg_real.py
rosrun imitation-learning-ros-package fgbg_actor.py
```
If everything goes according to plan, a console view should pop up with the life mask predictions as well as the waypoints.
In order to start the autonomous flight, you can either use the keyboard or the joystick interface to publish an emtpy message on the '/go' topic.
You can over take the experiments with publishing an empty message on the '/overtake' topic.

## Troubleshoot
Just email me on kkelchtermans AT gmail.com. Thanks!