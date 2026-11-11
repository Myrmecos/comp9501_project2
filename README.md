# Intro
This is the work repo for COMP9501 Group 12.

We will implement thermal forward and inverse model to learn hand 3D pose.

TAPOR Dataset: https://drive.google.com/file/d/1JmTD0N_9hdhCS8JOSd9t2bQ_NmB9BbdB/view;

TAP3D dataset: https://huggingface.co/datasets/TAP3DNow/TAP3D/tree/main

TAPOR code: https://github.com/aiot-lab/TAPOR

TAP3D code: https://github.com/aiot-lab/TAP3D




# Structure
- `config/`: folder for experiment configuration files, e.g. config.yml
- `data/`: data folder
- `log/`: training log are placed here
- `models/`: different model. We use one python file for one model.
- `weights/`: keep training weights
- `losses.py`: functions that define the loss for training.
- `metrics.py`: functions that define metrics for evaluation.
- `reproduce_results.ipynb`: demo script for quick start.
- `requirements.txt`: for installing python dependency.
- `utils.py`: stores utility functions.