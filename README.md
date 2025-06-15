# AAAI-2025 DrRL

This repository contains the code for the AAAI 2025 paper "Advancing Loss Functions in Recommender Systems: A Comparative Study with a Rényi Divergence-Based Solution".


# Conda Environment

We provide the environment requirements, you can construct the environment as:
```
conda env create -f environment.yaml
```

This will create a conda environment named `recommendation_benchmark`. To activate the environment, run:
```
conda activate rec_gym
```

## Code Structure

The code is organized as follows:

```
IR-Benchmark-SLatK
── README.md
├── dataset											# The dataloader
├── main.py											# The Trainer
├── mainAdvInfoNCE.py						# The Trainer for AdvInfoNCE
├── mainAdvInfoNCENoise.py			# The Trainer with False Negative Noise for AdvInfoNCE 
├── mainNoise.py								# The Trainer with False Negative Noise for others
├── model												# The backbone recommendation models
│   ├── model_MF.py							# The MF Model
│   ...
├── normal_data									# The dataset we used
│   └── IID_Data								# The IID dataset
│       ├── Gowalla							# The Gowalla dataset
│       │   ├── test.txt				# The test set
│       │   ├── train.txt				# The original train set
│       │   ├── train_data.txt	# The real reain set split from train.txt with 9:1
│       │   ├── transfer_train_valid.py	# The valid split script
│       │   └── valid_data.txt					# The valid data
│       ...
├── optimizer										# The loss function optimizer
│   ├── optim_DrRL.py						# The DrRL optimizer
│   ...								
└── tools												# Usefule tools
    ├── procedure.py						# The trainer for one epoch
    ├── world.py								# The hyperparameter/arguments						
    ...
```


## NNI Hyperparameter Tuning

We provide automatically conduct hyperparameter tuning with NNI framework. You can run:
```
nnictl create --config ./Config/TLatK.yaml --port [Your Possible Port]
```



## Citation

If you find this repository useful, please consider citing the following paper:

```bibtex
@inproceedings{zhang2025advancing,
  title={Advancing Loss Functions in Recommender Systems: A Comparative Study with a R{\'e}nyi Divergence-Based Solution},
  author={Zhang, Shengjia and Chen, Jiawei and Li, Changdong and Zhou, Sheng and Shi, Qihao and Feng, Yan and Chen, Chun and Wang, Can},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={39},
  number={12},
  pages={13286--13294},
  year={2025}
}
```