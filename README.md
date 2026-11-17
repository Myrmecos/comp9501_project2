# Intro
This is the work repo for COMP9501 Group 12.

We will implement thermal forward and inverse model to learn hand 3D pose.

TAPOR Training Dataset (pose Trainset): https://drive.google.com/file/d/12Ppoh15nomnVAKo2UFQRqgcLeDRh5hGk/view

TAPOR Test Dataset (pose Testset): https://drive.google.com/file/d/1cnwdelIiV4V15rq3ANN93YsNvAjpvV5O/view

TAPOR Gesture Recognition Dataset: https://drive.google.com/file/d/1JmTD0N_9hdhCS8JOSd9t2bQ_NmB9BbdB/view

TAP3D dataset: https://huggingface.co/datasets/TAP3DNow/TAP3D/tree/main

TAPOR code: https://github.com/aiot-lab/TAPOR

TAP3D code: https://github.com/aiot-lab/TAP3D

Local data (gitignored): `data/tapor/{Trainset,Testset,gesture}/`

Paper digests: `paper_report/tapor.md`, `paper_report/tap3d.md`

Dataset overview deck: `data_overview/index.html`

Dataset notebook: `visualize_dataset.ipynb`




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