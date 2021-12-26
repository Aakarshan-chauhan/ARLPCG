from mlagents_envs.side_channel import side_channel
import numpy as np
import torch 
import torch.nn as nn
from torch.distributions import Normal
from PythonCode.LoadingEnv import CustomEnv
from mlagents_envs.environment import UnityEnvironment
import gym
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
import wandb


class Agent(nn.Module):
    def __init__(self, observation_shape, action_shape, learning_rate = 0.0003):
        super(Agent, self).__init__()
        self.observation_dims = np.array(observation_shape).prod()
        self.action_dims = np.array(action_shape).prod()

        self.actor_means = nn.Sequential(
            nn.Linear(self.observation_dims, 128),
            nn.Tanh(),
            nn.Linear(128, 32),
            nn.Tanh(),
            nn.Linear(32, self.action_dims)
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, self.action_dims))

        self.critic = nn.Sequential(
            nn.Linear(self.observation_dims, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        self.optimizer= torch.optim.Adam(self.parameters(), lr = learning_rate)

    def get_values(self, observations):
        assert isinstance(observations, torch.Tensor)
        assert len(observations.shape) >1
        return self.critic(observations)

    def get_actions_and_values(self, observations, actions = None):
        means= self.actor_means(observations)
        log_stds = self.actor_logstd.expand_as(means)
        stds = torch.exp(log_stds)

        dist = Normal(means, stds)
        if actions is None:
            actions = dist.sample()

        return actions, dist.log_prob(actions).sum(1), dist.entropy().sum(1), self.get_values(observations)



def get_gae():
    next_value = agent.get_values(next_obs).reshape(1, -1)
    advantages = torch.zeros_like(rewards)
    lastgaelam = 0
    for t in reversed(range(buffer_size)):
        if t+1 == buffer_size:
            nextnonterminal = 1. - next_done
            nextvalues = next_value
        else:
            nextnonterminal = 1 - dones[t+1]
            nextvalues = values[t+1]
        delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
        advantages[t] = lastgaelam = delta + gamma * lamda * lastgaelam * nextnonterminal

    returns = advantages + values

    return returns, advantages

def update():
    binds = np.arange(buffer_size)
    np.random.shuffle(binds)
    l = []
    for idx in range(0, buffer_size, minibatch_size):
        start = idx 
        end = idx + minibatch_size
        mbinds = binds[start:end]

        mb_obs = observations[mbinds]
        mb_acts = actions[mbinds]
        mb_rets = returns[mbinds]
        mb_advs = advantages[mbinds]
        mb_dones = dones[mbinds]
        mb_logprobs = log_probs[mbinds]

        _, newlogprob, entropy, new_values = agent.get_actions_and_values(
            mb_obs, mb_acts
        )
        logratio = newlogprob - mb_logprobs
        ratio = logratio.exp()

        mb_advs = (mb_advs - mb_advs.mean())/(mb_advs.std() + 1e-8)

        loss1 = -1 * mb_advs * ratio
        loss2 = -1 * mb_advs * torch.clamp(ratio, 1 + clip, 1- clip)
        aloss = torch.max(loss1, loss2).mean()

        closs = .5 * torch.square(new_values - mb_rets).mean()
        eloss = entropy.mean()

        complete_loss = aloss + closs_coeff * closs - eloss_coeff * eloss

        agent.optimizer.zero_grad()
        complete_loss.backward()
        agent.optimizer.step()
        
import tqdm

if __name__=="__main__":
    closs_coeff = 0.5
    eloss_coeff = 0.2
    clip = 0.2
    gamma = .99
    lamda = .91
    wandb.init(project="MiniProject-PPO", monitor_gym=True)
    wandb.config.closs_coeff = closs_coeff
    wandb.config.eloss_coeff = eloss_coeff
    wandb.config.clip = clip
    wandb.config.gamma = gamma
    wandb.config.lamda = lamda
    wandb.config.time_scale = 5

    sc = EngineConfigurationChannel()
    sc.set_configuration_parameters(time_scale=5)

    print("Opening ENV")

    gym_id = r"D:\My C and Python Projects\2021\UnityStuff\MiniProject\Build_A NoRT DummyGen\env"
    uenv = UnityEnvironment(gym_id, side_channels = [sc])

    cenv = CustomEnv(uenv)
    total_timesteps = 50000


    # Number of minibatch updates after one rollout
    update_epochs = 10
    # Number of steps before the update
    buffer_size = 1024
    # size of the minibatches 
    minibatch_size = buffer_size//update_epochs
    
    steps_per_agent = total_timesteps//10 // buffer_size
    num_iters = total_timesteps // steps_per_agent
    agents = {}
    for name in cenv.b_names:
        agents[name] = Agent(cenv.observation_shapes[name], cenv.action_shapes[name])

    for n in range(total_timesteps):
        print(n)
        print(n//steps_per_agent)
        if n//steps_per_agent % 2!=0:
            train_name = cenv.b_names[1]
            infer_name = cenv.b_names[0]
            print("Training the Solver")
        else:
            train_name = cenv.b_names[0]
            infer_name = cenv.b_names[1]
            print("Training the Generator")
            continue 

        observations = torch.zeros((buffer_size, ) + cenv.observation_shapes[train_name])
        actions = torch.zeros((buffer_size, ) + cenv.action_shapes[train_name])
        rewards = torch.zeros((buffer_size, 1))
        dones = torch.zeros((buffer_size, 1))
        log_probs = torch.zeros((buffer_size, 1))
        values = torch.zeros((buffer_size, 1))

        cenv.reset()

        next_obs = torch.Tensor([cenv.observation[name]])
        next_done = torch.Tensor([cenv.done[name]])
        next_values = agents[train_name].get_values(next_obs)

        episode_reward = []
        for step in tqdm.trange(buffer_size):
            observations[step] = next_obs
            values[step] = next_values

            infer_obs = torch.Tensor([cenv.observation[infer_name]])

            with torch.no_grad():
                a, lp, _, next_values = agents[train_name].get_actions_and_values(next_obs)
                cenv.set_action(train_name, a.detach().cpu().numpy())

                dummy_a , _, __, ___ = agents[infer_name].get_actions_and_values(infer_obs)
                cenv.set_action(infer_name, dummy_a.detach().cpu().numpy())

                cenv.step()
                cenv.get_obs()
                
            
            next_obs = torch.Tensor([cenv.observation[train_name]])
            next_done = torch.Tensor([cenv.done[train_name]])
            
            dones[step] = next_done
            rewards[step] = torch.Tensor([cenv.reward[train_name]])
            log_probs[step] = lp.reshape(1, -1)
            actions[step] = a

            episode_reward.append(cenv.reward[train_name])
            if cenv.done[train_name] or cenv.done[infer_name]:
                cenv.reset()
               
                wandb.log({f"Rewards {train_name}" : np.sum(episode_reward)})
                
                next_obs = torch.Tensor([cenv.observation[train_name]])  
                next_values = agents[train_name].get_values(next_obs)
                episode_reward = []
                
        agent = agents[train_name]
        with torch.no_grad():
            returns, advantages = get_gae()

        log_probs = log_probs.reshape(-1)
        rewards = rewards.reshape(-1)
        returns = returns.reshape(-1)
        advantages = advantages.reshape(-1)
        
        print("updating")
        update()

        if n%20 == 0:
            torch.save(agents[train_name].state_dict(), r"Weights/ppo.pt")