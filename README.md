# **MENTOR: Mixture-of-Experts Network with Task-Oriented Perturbation for Visual Reinforcement Learning**

<p align="center" style="font-size: 50px">
   <a href="https://arxiv.org/abs/2410.14972">[Paper]</a>&emsp;<a href="https://suninghuang19.github.io/mentor_page/">[Project Website]</a>
</p>


This repository is the official PyTorch implementation of **MENTOR**. **MENTOR** is a highly efficient visual RL algorithm that excels in both simulation and real-world complex robotic learning tasks.


# 🛠️ Installation Instructions

First, create a virtual environment and install all required packages. 

```bash
sudo apt update
sudo apt install libosmesa6-dev libegl1-mesa libgl1-mesa-glx libglfw3 
conda env create -f conda_env.yml 
conda activate mentor
pip3 install torch==1.12.1+cu116 torchvision==0.13.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
```

Next, install the additional dependencies required for MetaWorld and Adroit. 

```
cd metaworld
pip install -e .
cd ..
cd rrl-dependencies
pip install -e .
cd mj_envs
pip install -e .
cd ..
cd mjrl
pip install -e .
```

Tips: please check that your mujoco_py can use gpu render to improve FPS during training.

```
mujoco_py.cymj
<module 'cymj' from './mujoco_py/generated/cymj_2.1.2.14_38_linuxgpuextensionbuilder_38.so'>
```

## 💻 Code Usage

If you would like to run MENTOR on [DeepMind Control Suite](https://github.com/google-deepmind/dm_control), please use train_dmc.py to train MENTOR policies on different configs.

```bash
python train_dmc.py task=dog_walk agent=mentor
```

If you would like to run MENTOR on [MetaWorld](https://meta-world.github.io/), please use train_mw.py to train MENTOR policies on different configs.

```bash
python train_mw.py task=coffee-push agent=mentor_mw
```

If you would like to run MENTOR on Adroit, please use train_adroit.py to train MENTOR policies on different configs.

```bash
python train_adroit.py task=pen agent=mentor_adroit
```

You can also specify the configs of MoE and task-oriented perturbation in MENTOR using following command:

```bash
python train_dmc.py task=dog_walk agent=mentor agent.moe_gate_dim=256 agent.moe_hidden_dim=256 agent.tp_set_size=16
```

## 📝 Citation

If you use our method or code in your research, please consider citing the paper as follows:

```
@article{huang2024mentor,
  title={MENTOR: Mixture-of-Experts Network with Task-Oriented Perturbation for Visual Reinforcement Learning},
  author={Huang, Suning and Zhang, Zheyu and Liang, Tianhai and Xu, Yihan and Kou, Zhehao and Lu, Chenhao and Xu, Guowei and Xue, Zhengrong and Xu, Huazhe},
  journal={arXiv preprint arXiv:2410.14972},
  year={2024}
}
```

## 🙏 Acknowledgement

MENTOR is licensed under the MIT license. MuJoCo and DeepMind Control Suite are licensed under the Apache 2.0 license. We would like to thank [DrM](https://github.com/XuGW-Kevin/DrM) and [DrQv2](https://github.com/facebookresearch/drqv2) authors for open-sourcing the codebase. Our implementation builds on top of their repository.
