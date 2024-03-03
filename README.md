# Offline Multi-Agent Reinforcement Learning with Implicit Global-to-Local Value Regularization (NeurIPS 2023)
The official implementation of "[Offline Multi-Agent Reinforcement Learning with Implicit Global-to-Local Value Regularization](https://arxiv.org/abs/2307.11620)". OMIGA provides a principled framework to convert global-level value regularization into equivalent implicit local value regularizations and simultaneously enables in-sample learning, thus elegantly bridging multi-agent value decomposition and policy learning with offline regularizations. This repository is inspired by the [TRPO-in-MARL](https://github.com/cyanrain7/TRPO-in-MARL) library for online Multi-Agent RL.

**This repo provides the implementation of OMIGA in Multi-agent MuJoCo and SMAC**

## Branches Overview
| Branch name 	| Usage 	|
|:---:	|:---:	|
| [master](https://github.com/ZhengYinan-AIR/OMIGA) 	| OMIGA implementation for ``Multi-agent MujoCo``. |
| [SMAC](https://github.com/ZhengYinan-AIR/OMIGA/tree/SMAC) 	| OMIGA implementation for ``SMAC``. 	|


## Installation
``` Bash
conda create -n env_name python=3.9
conda activate OMIGA
git clone https://github.com/ZhengYinan-AIR/OMIGA.git
cd OMIGA
pip install -r requirements.txt
```

## How to run

Before running the code, you need to download the necessary offline datasets ([Download link](https://cloud.tsinghua.edu.cn/d/dcf588d659214a28a777/)). Then, make sure the config file at [configs/config.py](https://github.com/ZhengYinan-AIR/Offline-MARL/blob/master/configs/config.py) is correct. Set the **data_dir** parameter as the storage location for the downloaded data, and configure parameters **scenario**, **agent_conf**, and **data_type**. You can run the code as follows:
``` Bash
# If the location of the dataset is at: "/data/Ant-v2-2x4-expert.hdf5"
cd OMIGA
python run_mujoco.py --data_dir="/data/" --scenario="Ant-v2" --agent_conf="2x4" --data_type="expert"
```




## Bibtex
If you find our code and paper can help, please cite our paper as:
```
@article{wang2023offline,
  title={Offline Multi-Agent Reinforcement Learning with Implicit Global-to-Local Value Regularization},
  author={Wang, Xiangsen and Xu, Haoran and Zheng, Yinan and Zhan, Xianyuan},
  journal={Advances in Neural Information Processing Systems},
  year={2023}
}
```
