# What I've changed
I mainly changed model/point.py file for 'Adaptive K-Loss'.


# Training
python launch.py --config configs/nerfies.yaml datapipeline.dataset.data_path='path/to/nerfies/dataset'

# If you ensure 'data_path' in YAML file, then run this code for training
python launch.py --config configs/nerfies.yaml
python launch.py --config configs/dnerf.yaml


# **Datasets** : D-NeRF Datasets (https://github.com/albertpumarola/D-NeRF)
If you want to use D-NeRF Datasets, use configs/dnerf.yaml file. (Check 'data_path' in it.)

# For runing this repository, you need..
datasets
pointrix
...
