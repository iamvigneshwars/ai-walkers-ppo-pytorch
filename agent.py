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
        self.writer = SummaryWriter()
        self.log_probs = []
        self.values = []
        self.states = []
        self.actions = []
        self.rewards = []
        self.masks = []

    def make_env(self):
        def thunk():
            env = gym.make(self.env_id)
            return env
        return thunk

    def calculate_gae(self,next_value,gamma, lmda):
        values =  self.values + [next_value]
        gae= 0
        returns = []
        for step in reversed(range(len(self.rewards))):
            delta = self.rewards[step] + gamma * values[step + 1] * self.masks[step] - values[step]
            gae = delta + gamma * lmda * self.masks[step] * gae
            returns.append(gae + values[step])
        return list(reversed(returns))

    def normalize(self, x):
        x -= x.mean()
        x /= (x.std() + 1e-8)
        return x

    def sample_batch(self, states, actions, log_probs, returns, advantages):
        batch_size = states.size(0)

        for _ in range(batch_size // args.mini_batch):
            rand_ids = np.random.randint(0, batch_size , args.mini_batch)
            yield states[rand_ids, :], actions[rand_ids, :],log_probs[rand_ids, :], \
                    returns[rand_ids, :], advantages[rand_ids, :]


    def learn(self):
        envs = [self.make_env() for i in range(args.n_workers)]
        envs = SubprocVecEnv(envs)
        env = gym.make(self.env_id)
        num_inputs = env.observation_space.shape[0]
        num_outputs = env.action_space.shape[0]
        model = PPO(num_inputs, num_outputs).to(self.device)

        if (args.load):
            model.load_state_dict(torch.load(args.model))
        optimizer = optim.Adam(model.parameters(), lr = args.lr)

        frame_idx = 0
        train_epoch = 0
        best_reward = None

        state = envs.reset()
        early_stop = False

        while not early_stop:

            for _ in range(args.ppo_steps):
                state = torch.FloatTensor(state).to(device)
                dist, value = model(state)
                action = dist.sample()
                next_state, reward, done, _ = envs.step(action.cpu().numpy())
                log_prob = dist.log_prob(action)
                
                self.log_probs.append(log_prob)
                self.values.append(value)
                self.rewards.append(torch.tensor(reward, dtype = torch.float32).unsqueeze(1).to(self.device))
                self.masks.append(torch.tensor(1 - done).unsqueeze(1).to(self.device))
                self.states.append(state)
                self.actions.append(action)

                state = next_state
                frame_idx += 1

            next_state = torch.FloatTensor(next_state).to(device)
            _, next_value = model(next_state)
            returns = self.calculate_gae(next_value, args.gamma, args.lmda)

            returns = torch.cat(returns).detach()
            log_probs = torch.cat(self.log_probs).detach()
            values = torch.cat(self.values).detach()
            states = torch.cat(self.states)
            actions = torch.cat(self.actions)
            advantage = returns - values
            advantage = self.normalize(advantage)

            for _ in range(args.epochs):
                for st, actn, old_log_probs, retn, adv in \
                        self.sample_batch(states, actions, log_probs, returns, advantage):

                    dist, value = model(st)
                    entropy = dist.entropy().mean()
                    new_log_probs = dist.log_prob(actn)
                    ratio = (new_log_probs - old_log_probs).exp()
                    surr1 = ratio * adv
                    surr2 = torch.clamp(ratio, 1.0- args.epsilon, 1.0 + args.epsilon) * adv
                    actor_loss = - torch.min(surr1, surr2).mean()
                    critic_loss = (retn  - value).pow(2).mean()
                    total_loss = args.c1  * critic_loss + actor_loss - args.c2 * entropy

                    optimizer.zero_grad()
                    total_loss.backward()
                    optimizer.step()

            train_epoch += 1
            self.states.clear()
            self.actions.clear()
            self.rewards.clear()
            self.masks.clear()
            self.log_probs.clear()
            self.values.clear()

            if train_epoch % args.epochs == 0:
                test_reward = np.mean([self.play(env, model) for _ in range(10)])
                self.writer.add_scalar('test_rewards', test_reward, frame_idx)
                print('Frame %s. reward : %s ' % (frame_idx, test_reward))
                # if best_reward is None or best_reward <= test_reward:
                #     if best_reward is not None:
                #         print("Best reward updated : %.3f -> %.3f" % (best_reward, test_reward))
                #         name = "%s_best_%+.3f_%d.pth" %(self.env_id, test_reward, frame_idx)
                #         fname = os.path.join('.', 'checkpoints', name)
                #         torch.save(model.state_dict(), fname)
                #     best_reward = test_reward
            
    
    def play(self,env = None, model = None, human = False):

        if not env:
            env = gym.make(self.env_id)
            env = gym.wrappers.Monitor(env, './', force = True)

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
            dist, _ = model(state)
            action = dist.sample().cpu().numpy()[0]
            next_state, reward, done, _ = env.step(action)
            state = next_state
            total_reward += reward
            if (human):
                print("***SCORE :", total_reward,"***")
        return total_reward

        
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
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
    parser.add_argument("-epsilon", help = "entropy beta", default = 0.02, type = float)
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    agent = Agent(args.env, device)

    if (args.learn):
        agent.learn()
    if (args.play):
        agent.play(human = True)
