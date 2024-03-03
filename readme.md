# Offline Multi-Agent Reinforcement Learning with Implicit Global-to-Local Value Regularization (NeurIPS 2023)


## Installing StarCraft II and SMAC

Set up StarCraft II (2.4.10) and SMAC using the following command.  Alternatively, you could install them manually, following the official link: https://github.com/oxwhirl/smac.

```
wget https://blzdistsc2-a.akamaihd.net/Linux/SC2.4.10.zip
unzip -P iagreetotheeula SC2.4.10.zip

wget https://github.com/oxwhirl/smac/releases/download/v0.1-beta1/SMAC_Maps.zip
unzip SMAC_Maps.zip
mv SMAC_Maps ./StarCraftII/Maps

export SC2PATH=$(pwd)/StarCraftII
pip install git+https://github.com/oxwhirl/smac.git
rm -rf SC2.4.10.zip SMAC_Maps.zip
```

## Offline data

The offline SMAC dataset we used is provided by paper “[Offline Pre-trained Multi-Agent Decision Transformer: One Big Sequence Model Tackles All SMAC Tasks](https://arxiv.org/abs/2112.02845v3)". For each original large dataset, we randomly sample 1000 episodes as our dataset ([Download link](https://cloud.tsinghua.edu.cn/d/f3c509d7a9d54ccd89c4/)). 



## How to run

Before running the code, you need to make sure the config in **run_omiga_sc2.py**  is correct. Set the **offline_data_dir**  parameter as the storage location for the downloaded data, and configure parameters **map_name** to ensure its alignment with data. You can run the code as follows:

```
python run_omiga_sc2.py  --offline_data_dir='../data/6h_vs_8z/good/' --map_name='6h_vs_8z'
```
