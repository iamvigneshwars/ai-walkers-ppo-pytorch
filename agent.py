import gym
from gym.wrappers.monitoring.video_recorder import VideoRecorder
import time
import os
import pybulletgym
import torch
from model import PPO
from torch.distributions import Normal
import torch.optim as optim
from tensorboardX import SummaryWriter
import numpy as np
from multiprocessing_env import SubprocVecEnv
import argparse
import pybullet_envs as pe

class Agent:
    def __init__(self, environment, device):
        self.env_id = environment
        self.device = device
        self.writer = SummaryWriter(f"runs/{args.exp}")

    def make_env(self):
        def thunk():
            env = gym.make(self.env_id)
            return env
        return thunk

    def calculate_gae(self, next_value, rewards, masks, values,gamma = 0.99,lmda = 0.95 ):
        values = values + [next_value]
        gae = 0
        returns = []
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
            gae = delta + gamma * lmda * masks[step] * gae
            returns.insert(0, gae + values[step])
        return returns

    def normalize(self, x):
        x -= x.mean()
        x /= (x.std() + 1e-8)
        return x

    def learn(self):
        
        envs = [self.make_env() for i in range(args.n_workers)]
        envs = SubprocVecEnv(envs)
        env = gym.make(self.env_id)
        num_inputs = env.observation_space.shape[0]
        num_outputs = env.action_space.shape[0]
        model = PPO(num_inputs, num_outputs).to(self.device)
        optimizer = optim.Adam(model.parameters(), lr = args.lr)

        if (args.load):
            model.load_state_dict(torch.load(args.model))

        global_steps  = 0
        best_reward = None
        state = envs.reset()
        early_stop = False
        train_epoch = 0
        TARGET_REWARD = 2500

        while not early_stop:

            log_probs = []
            values    = []
            states    = []
            actions   = []
            rewards   = []
            masks     = []

            for _ in range(args.ppo_steps):
                state = torch.FloatTensor(state).to(self.device)
                with torch.no_grad():
                    dist, value = model(state)

                action = dist.sample()
                next_state, reward, done, info = envs.step(action.cpu().numpy())
                log_prob = dist.log_prob(action)

                log_probs.append(log_prob)
                values.append(value)
                rewards.append(torch.tensor(reward, dtype=torch.float32).unsqueeze(1).to(self.device))
                masks.append(torch.tensor(1 - done, dtype=torch.float32).unsqueeze(1).to(self.device))

                states.append(state)
                actions.append(action)

                state = next_state
                global_steps += 1
                
                for item in info:
                    if "episode" in item.keys():
                        print(f"global_step={global_steps}, episodic_return={item['episode']['r']}")
                        writer.add_scalar("charts/episodic_return", item["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", item["episode"]["l"], global_step)
                        break
                
            next_state = torch.FloatTensor(next_state).to(self.device)
            with torch.no_grad():
                _, next_value = model(next_state)
            returns = self.calculate_gae(next_value, rewards, masks, values)

            returns   = torch.cat(returns).detach()
            log_probs = torch.cat(log_probs).detach()
            values    = torch.cat(values).detach()
            states    = torch.cat(states)
            actions   = torch.cat(actions)
            advantage = returns - values
            advantage = self.normalize(advantage)

            b_inds = np.arange(states.size(0))
            for _ in range(args.epochs):
                np.random.shuffle(b_inds)
                for start in range(0, states.size(0), args.mini_batch):
                    end = start + args.mini_batch
                    mb_inds = b_inds[start:end]

                    dist, value = model(states[mb_inds, :])
                    entropy = dist.entropy().mean()
                    new_log_probs = dist.log_prob(actions[mb_inds, :])
                    ratio = (new_log_probs - log_probs[mb_inds, :]).exp()
                    p_loss1 = ratio * advantage[mb_inds, :]
                    p_loss2 = torch.clamp(ratio, 1.0 - args.epsilon, 1.0 + args.epsilon) * advantage[mb_inds, :]
                    p_loss = - torch.min(p_loss1, p_loss2).mean()
                    v_loss = (returns[mb_inds, :] - value).pow(2).mean()
                    loss = args.c1 * v_loss + p_loss - args.c2 * entropy

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            self.writer.add_scalar("losses/policy_loss", p_loss.item(), global_steps)
            self.writer.add_scalar("losses/value_loss", v_loss.item(), global_steps)
            self.writer.add_scalar("losses/total", loss.item(), global_steps)
            train_epoch +=1

            if train_epoch % args.epochs == 0:
                test_reward = np.mean([self.play(env, model, device) for _ in range(10)])
                self.writer.add_scalar("test_rewards", test_reward, global_steps)
                print('Frame %s. reward: %s' % (global_steps, test_reward))
                if best_reward is None or best_reward < test_reward:
                    if best_reward is not None:
                        print("Best reward updated: %.3f -> %.3f" % (best_reward, test_reward))
                        name = "%s_score_%+d_%d.pth" % (self.env_id, test_reward, global_steps)
                        fname = os.path.join('.', 'checkpoints', name)
                        torch.save(model.state_dict(), fname)
                    best_reward = test_reward
                if test_reward > TARGET_REWARD: early_stop = True
    
    # def test_env(self, env, model, device, deterministic=True):
    #     state = env.reset()
    #     done = False
    #     total_reward = 0
    #     while not done:
    #         state = torch.FloatTensor(state).unsqueeze(0).to(device)
    #         dist, _ = model(state)
    #         action = dist.mean.detach().cpu().numpy()[0] if deterministic \
    #             else dist.sample().cpu().numpy()[0]
    #         next_state, reward, done, _ = env.step(action)
    #         state = next_state
    #         total_reward += reward
    #     return total_reward

    def play(self,env = None, model = None, human = False):

        if not env:
            env = gym.make(self.env_id)
            env = gym.wrappers.Monitor(env, "./videos")

        if not model:
            model = PPO(env.observation_space.shape[0], env.action_space.shape[0]).to(self.device)
            model.load_state_dict(torch.load(args.model))
        
        if human:
            env.render()

        state = env.reset()
        done= False
        total_reward = 0
        while not done:
            state = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                dist, _ = model(state)
            # action = dist.sample().cpu().numpy()[0]
            action = dist.mean.detach().cpu().numpy()[0]
            next_state, reward, done, _ = env.step(action)
            state = next_state
            total_reward += reward
            if (human):
                print("***SCORE :", total_reward,"***")
        return total_reward

        
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", help = "Name of the experiment",type=str, default = "PPO" )
    parser.add_argument("--env", help = "OpenAI gym environment", default = "HalfCheetahPyBulletEnv-v0", type = str)
    parser.add_argument("--learn", help = "Agent starts to learn",  action= 'store_true')
    parser.add_argument("--play", help = "Agent starts to play", action= 'store_true')
    parser.add_argument("-n_workers", help = "Number of environments", default = 8, type = int)
    parser.add_argument("-mini_batch", help = "Size of mini batch to sample", default = 64, type = int)
    parser.add_argument("-lr", help = "Model learning rate", default = 1e-4, type = float)
    parser.add_argument("-gamma", help = "return discount factor", default = 0.99, type = float)
    parser.add_argument("-lmda", help = "gae lambda", default = 0.95, type = float)
    parser.add_argument("-epochs", help = "number of updates", default = 10, type = int)
    parser.add_argument("-model", help = "pretrained model", type = str)
    parser.add_argument("-load", help = "load checkpoint", action = 'store_true')
    parser.add_argument("-ppo_steps", help = "Number of steps before update", default = 256, type = int)
    parser.add_argument("-c1", help = "critic discount", default = 0.5, type = float)
    parser.add_argument("-c2", help = "entropy beta", default = 0.001, type = float)
    parser.add_argument("-epsilon", help = "entropy beta", default = 0.2, type = float)
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    agent = Agent(args.env, device)

    if (args.learn):
        agent.learn()
    if (args.play):
        agent.play(human = True)
