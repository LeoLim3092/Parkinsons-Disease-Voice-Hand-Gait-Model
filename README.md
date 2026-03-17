# Parkinsons Disease: Voice, Hand, and Gait Models
## Create virtual environment
1. Install conda and create conda environment 
	conda create -n "pdmodel" python=3.8
2. run conda environment
	conda activate pdmodel

## Install package
- install lastest visual studio (for window only) https://visualstudio.microsoft.com/downloads/
- pip install -r ./requirement.txt
- install pytorch
	- conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
- install mmpose 
	- pip install -U openmim
	- mim install mmengine
	- mim install mmcv-full==1.7.0
	- mim install "mmdet==2.28.1"
	- mim install mmpose==0.29.0 --user

## Settings
- Update new trained models in the Pre-trained models folder with all folds models

## Features
- Voice features extraction
	- voiceFeatureExtraction.py
- Hand features extraction
	- handFeatrureExtraction.py
- Gait features extraction
	- gaitFeatureExtraction.py
- Deploy trained model prediction
	- deployModel.py
