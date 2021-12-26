from mlagents_envs.environment import BaseEnv
from mlagents_envs.base_env import ActionTuple
import torch
class CustomEnv():
    def __init__(self, env : BaseEnv):
        self.env = env
        self.env.reset()
        self.b_names = list(env.behavior_specs.keys())
        
        self.observation_shapes = {}
        self.observation = {}
        self.reward = {}
        self.done = {}
        self.actions = {}
        self.action_shapes = {}
        for name in self.b_names:
            obs_shape = self.env.behavior_specs[name].observation_specs[0].shape
            self.observation_shapes[name] = obs_shape
            self.observation[name] = torch.zeros(obs_shape)
            self.action_shapes[name] = (self.env.behavior_specs[name].action_spec.continuous_size, )
            self.done[name] = False
            
    def reset(self):
        self.env.reset()
        self.get_obs()
        return self.observation

    def set_action(self, name, action):
        at = ActionTuple()
        at.add_continuous(action)
        
        _, ts= self.env.get_steps(name)
        if len(ts)>0:
            return

        self.env.set_actions(name, at)
    
    def step(self):
        self.env.step()
        self.get_obs()
        return self.observation, self.reward, self.done

    def get_obs(self):
        for name in self.b_names:
            ds, ts = self.env.get_steps(name)
            if len(ds) >0:
                for id in ds:
                    self.observation[name] = ds[id].obs[0]
                    self.reward[name] = ds[id].reward
                    self.done[name] = False
            else:
                for id in ts:
                    self.observation[name] = ts[id].obs[0]
                    self.reward[name] = ts[id].reward
                self.done[name] = True

    def close(self):
        self.env.close()

    def __str__(self):
        return str(self.b_names)