**This repo is based on 'Gaussian-Flow'**  
Paper : https://arxiv.org/abs/2312.03431  
Github : https://github.com/NJU-3DV/Gaussian-Flow  

# What I've changed
I mainly changed model/point.py file for 'Adaptive K-Loss'.


# Training
python launch.py --config configs/nerfies.yaml datapipeline.dataset.data_path='path/to/nerfies/dataset'

# If you ensure about 'data_path' in YAML file, then run this code for training
python launch.py --config configs/nerfies.yaml  
python launch.py --config configs/dnerf.yaml


# Datasets : D-NeRF Datasets 
Download D-NeRF Datasets : https://github.com/albertpumarola/D-NeRF  
If you want to use D-NeRF Datasets, use configs/dnerf.yaml file. (Check 'data_path' in it.)

# For runing this repository, you need..
Check the version of libraries (following requirements.txt)  
Download datasets & pointrix  
Make a 'outputs' folder to save the outputs  
...
