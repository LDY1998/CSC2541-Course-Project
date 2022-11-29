import os
import json
import pickle
import numpy as np
import load_expert_policy
import torch
from create_env import create_env

EXPERT_POLICY_PATH = "./experts/Hopper-v2.pkl"
ENV_NAME = "Hopper-v2"

def run_expert(num_traj=1, max_timesteps=1000):
    

    print('loading and building expert policy')
    policy_fn = load_expert_policy.ExpertPolicy(EXPERT_POLICY_PATH)
    print('loaded atraintnd built')
    
    save_name = ENV_NAME # + '_' + time.strftime('%Y-%m-%d-%H-%M-%S')
 

    import gym
    env = gym.make(ENV_NAME)
    # env = create_env(random_noise=0, max_steps=max_timesteps)
    max_steps = max_timesteps or env.spec.timestep_limit

    returns = []
    observations = []
    actions = []
    timesteps = []
    rollouts = []
    for i in range(num_traj):
        timesteps_rollout = []
        print('iter', i)
        obs = env.reset()
        done = False
        totalr = 0.
        steps = 0
        while not done:
            x = torch.Tensor(obs[None,:])
            action = policy_fn(x)
            action = action.detach().numpy()
            observations.append(obs)
            actions.append(action)
            rollouts.append(i)
            timesteps_rollout.append(steps)
            obs, r, done, _ = env.step(action)
            totalr += r
            steps += 1
            # if render:
                # env.render()
            if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
            if steps >= max_steps:
                break
        returns.append(totalr)
        timesteps.extend(timesteps_rollout)

    print('returns {}'.format(returns))
    print('mean return {}'.format(np.mean(returns)))
    print('std of return {}'.format(np.std(returns)))
    
    with open(os.path.join('expert_data', save_name + '.json'), 'w') as f:
        json.dump({'returns': returns, 
                    'mean_return': np.mean(returns),
                    'std_return': np.std(returns)}, f)
    
    expert_data = {'observations': np.array(observations),
                    'actions': np.squeeze(np.array(actions), axis=1), "timesteps": np.array(timesteps), "trajectories": np.array(rollouts)}


    with open(os.path.join('expert_data', save_name + '.pkl'), 'wb') as f:
        pickle.dump(expert_data, f, pickle.HIGHEST_PROTOCOL)



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    # parser.add_argument('expert_policy_file', type=str)
    # parser.add_argument('envname', type=str)
    parser.add_argument('--render', action='store_true')
    parser.add_argument("--max_timesteps", type=int, default=1000)
    parser.add_argument('--num_traj', type=int, default=10,
                        help='Number of expert roll outs')
    args = parser.parse_args()
    run_expert(args.num_traj, args.max_timesteps)