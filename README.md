# ti5robot-15.53

Name: ti5robot-15.53
Version: 1.0.0
Summary: mujoco environments for humanoid robot
Author: Huiyang Kong
Author-email: huiyang.kong@ti5robot.com
License: BSD-3-Clause

Code for the papers: 
- [**Learning Bipedal Walking On Planned Footsteps For Humanoid Robots**](https://arxiv.org/pdf/2207.12644.pdf) (Humanoids2022) 
[Rohan P. Singh](https://rohanpsingh.github.io), [Mehdi Benallegue](https://unit.aist.go.jp/jrl-22022/en/members/member-benalleguem.html), [Mitsuharu Morisawa](https://unit.aist.go.jp/jrl-22022/en/members/member-morisawa.html), [Rafael Cisneros](https://unit.aist.go.jp/jrl-22022/en/members/member-cisneros.html), [Fumio Kanehiro](https://unit.aist.go.jp/jrl-22022/en/members/member-kanehiro.html)

- [**Learning Bipedal Walking for Humanoids with Current Feedback**](https://arxiv.org/pdf/2303.03724.pdf) (arxiv) 
[Rohan P. Singh](https://rohanpsingh.github.io), [Zhaoming Xie](https://zhaomingxie.github.io/), [Pierre Gergondet](https://unit.aist.go.jp/jrl-22022/en/members/member-gergondet.html), [Fumio Kanehiro](https://unit.aist.go.jp/jrl-22022/en/members/member-kanehiro.html) 
(WIP on branch `topic/omnidirectional-walk`)

## Code structure:
A rough outline for the repository that might be useful for adding your own robot:
```
ti5robot-15.53/
├── envs/                <-- Actions and observation space, PD gains, simulation step, control decimation, init, ...
├── tasks/               <-- Reward function, termination conditions, and more...
├── rl/                  <-- Code for PPO, actor/critic networks, observation normalization process...
├── models/              <-- MuJoCo model files: XMLs/meshes/textures
├── trained/             <-- Contains pretrained model for JVRC
└── scripts/             <-- Utility scripts, etc.
```

## Requirements:
- Python version: 3.10.6
- [Pytorch](https://pytorch.org/)
- pip install:
  - mujoco==3.2.6
  - mujoco-python-viewer==0.1.4
  - ray==2.40.0
  - transforms3d==0.4.2
  - matplotlib==3.9.3
  - scipy==1.14.1
  - torch==2.5.1

## Usage:

Environment names supported: 

| Task Description      | Environment name |
| ----------- | ----------- |
| Basic Walking Task   | 'jvrc_walk' |
| Stepping Task (using footsteps)  | 'jvrc_step' |


#### **To train:** 

```
$ python run_experiment.py train --logdir <path_to_exp_dir> --num_procs <num_of_cpu_procs> --env <name_of_environment>
```  

#### **To play:**

```
$ PYTHONPATH=.:$PYTHONPATH python scripts/debug_stepper.py --path <path_to_exp_dir>
```


#### **What you should see:**

*Ascending stairs:*  
![climb_up](https://user-images.githubusercontent.com/16384313/180697513-25796b1a-87e0-4ab2-9e5f-d86c58ebea36.gif)

*Descending stairs:*  
![climb_down](https://user-images.githubusercontent.com/16384313/180697788-d1a2eec0-0d3d-451a-95e0-9f0e60191c34.gif)

*Walking on curves:*  
![curve](https://user-images.githubusercontent.com/16384313/180697266-7b44beb3-38bf-4494-b568-963919dc1106.gif)


