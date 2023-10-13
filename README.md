# Offline Multi-Agent Reinforcement Learning with Implicit Global-to-Local Value Regularization (NeurIPS 2023)
The official implementation of "[Offline Multi-Agent Reinforcement Learning with Implicit Global-to-Local Value Regularization](https://arxiv.org/abs/2307.11620)". This repository is inspired by the [TRPO-in-MARL](https://github.com/cyanrain7/TRPO-in-MARL) library for online Multi-Agent RL.
## Installation
### Create environments
``` Bash
conda create -n env_name python=3.9
conda activate OMIGA
git clone https://github.com/ZhengYinan-AIR/OMIGA.git
cd OMIGA
pip install -r requirements.txt
```

### Multi-Agent MuJoCo
Following the instructios in https://github.com/openai/mujoco-py and https://github.com/schroederdewitt/multiagent_mujoco to setup a mujoco environment. In the end, remember to set the following environment variables:
``` Bash
LD_LIBRARY_PATH=${HOME}/.mujoco/mujoco210/bin
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so
```

## How to run

Before running the code, you need to download the necessary offline datasets ([Download link](https://cloud.tsinghua.edu.cn/d/dcf588d659214a28a777/)). Then, make sure the config file at [configs/config.py](https://github.com/ZhengYinan-AIR/Offline-MARL/blob/master/configs/config.py) is correct. Set the **data_dir** parameter as the storage location for the downloaded data, and configure parameters **scenario**, **agent_conf**, and **data_type**. You can run the code as follows:
``` Bash
# The location of the dataset is at: "/data/Ant-v2-2x4-expert.hdf5"
cd OMIGA
python run_mujoco.py --data_dir="/data/" --scenario="Ant-v2" --agent_conf="2x4" --data_type="expert"
```

## Weights and Biases Online Visualization Integration
This codebase can also log to [W&B online visualization platform](https://wandb.ai/site). To log to W&B, you first need to set your W&B API key environment variable:
```
wandb online
export WANDB_API_KEY='YOUR W&B API KEY HERE'
```
Then you can run experiments with W&B logging turned on:
```
python run_mujoco.py --wandb=True
```


## Bibtex
If you find our code and paper can help, please cite our paper as:
```
@article{wang2023offline,
  title={Offline Multi-Agent Reinforcement Learning with Implicit Global-to-Local Value Regularization},
  author={Wang, Xiangsen and Xu, Haoran and Zheng, Yinan and Zhan, Xianyuan},
  journal={arXiv preprint arXiv:2307.11620},
  year={2023}
}
```
