import gym
import pybullet
import torch
from model import PPO
from torch.distributions import Normal
import torch.optim as optim
from tensorboardX import SummaryWriter
import argparse

class Agent:
    def __init__(self, environment, device):
        self.env_id = environment
        self.device = device
        self.writter = SummaryWriter()
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

    def calculate_gae(self,next_value, gamma, lmda):
        values =  values + [next_value]
        gae= 0
        returns = []
        for step in reversed(range(len(self.rewards))):
            delta = self.rewards[step] + gamma * self.values[step + 1] * self.masks[step] - self.values[step]
            gae = delta + gamma * lmda * self.masks[step] * gae
            returns.append(gae + self.values[step])
        return reversed(returns)

    def normalize(x):
        x -= x.mean()
        x /= (x.std() + 1e-8)
        return x

    def sample_batch(self, returns, advantage):
        batch_size = states.size(0)

        for _ in range(batch_size // args.mini_batch):
            rand_ids = np.random.randint(0, batch_size , args.mini_batch)
            yield self.states[rand_ids, :], self.actions[rand_ids, :], self.log_probs[rand_ids, :], \
                    returns[rand_ids, :], advantage[rand_ids, :]


    def learn(self):
        envs = gym.vector.SyncVectorEnv([make_env() for i in range(args.n_workers)])
        num_inputs = envs.observation_space.shape[1]
        num_outputs = envs.action_space[0].shape[0]

        model = PPO(num_inputs, num_outputs).to(self.device)

        print(num_inputs)


        
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", help = "OpenAI gym environment", required = True, type = str)
    parser.add_argument("-n_workers", help = "Number of environments", default = 8, type = int)
    parser.add_argument("-mini_batch", help = "Size of mini batch to sample", default = 64, type = int)
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    agent = Agent(args.env, device)
    
    print(args.mini_batch)
