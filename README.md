# AI-Walkers-ppo-pytorch

Pytorch implimentation of proximal policy optimization with clipped objective function and generalized advantage estimation. The model is trained on Humanoid, hopper, ant and halfcheetah pybullet environment. To run multiple environments in multiple threads, SubprocVecEnv class from stable baseline is used (file included).

## Usage
Important command line arguments : <br>
`--env` environment name (note : works only for continuous pybullet environments) <br>
`--learn` agent starts training <br>
`--play` agent plays using pretrained model <br>
`-n_workers` number of environments <br>
`-load` continues training from given checkpoint <br>
`-model` load the model or checkpoint <br>
`-ppo_steps` number of steps before update <br>
`-epochs` number of updates <br>
`-mini_batch` batch size during ppo update <br>
`-lr` policy and critic learning rate <br>
`-c1` critic discount <br>
`-c2` entropy beta <br>

To train the agent:
```
# train new agent
python agent.py --learn --env <ENV_ID> 

# load checkpoints
python agent.py --learn --env <ENV_ID> -load -model <CHECKPOINT PATH> 

```
To Play: 
```
python agent.py --play --env <ENV_ID> -model <MODEL PATH>

```

| HumanoidBulletEnv-v0  | CheetahBulletEnv-v0 |
| :-------------------------:|:-------------------------: |
| ![](https://github.com/iamvigneshwars/ai-walkers-ppo-pytorch/blob/main/humanoid.gif) |  ![](https://github.com/iamvigneshwars/ai-walkers-ppo-pytorch/blob/main/cheetah.gif) |
| HopperBulletEnv-v0  | AntBulletEnv-v0 |
| ![](https://github.com/iamvigneshwars/ai-walkers-ppo-pytorch/blob/main/hopper.gif) |  ![](https://github.com/iamvigneshwars/ai-walkers-ppo-pytorch/blob/main/ant.gif) |


## Requirements

- [Python](https://www.python.org/downloads/) >= 3.7
- [Pytorch](https://pytorch.org/) >= 1.3.1
- [gym](https://gym.openai.com/)
- [pybullet](https://pybullet.org/wordpress/)
