import logging
import os
import json
import pickle
import numpy as np
import load_policy_torch
import torch


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('expert_policy_file', type=str)
    parser.add_argument('envname', type=str)
    parser.add_argument('--render', action='store_true')
    parser.add_argument("--max_timesteps", type=int)
    parser.add_argument('--num_rollouts', type=int, default=20,
                        help='Number of expert roll outs')
    args = parser.parse_args()

    print('loading and building expert policy')
    # policy_fn = load_policy.load_policy(args.expert_policy_file)
    policy_fn = load_policy_torch.ExpertPolicy(args.expert_policy_file)
    print('loaded atraintnd built')
    
    #logging.basicConfig(level=logging.INFO)
    save_name = args.envname # + '_' + time.strftime('%Y-%m-%d-%H-%M-%S')
    # file_handler = logging.FileHandler(os.path.join('expert_data', save_name + '.txt'))
    # logging.getLogger().addHandler(file_handler)
    

    import gym
    env = gym.make(args.envname)
    max_steps = args.max_timesteps or env.spec.timestep_limit

    returns = []
    observations = []
    actions = []
    for i in range(args.num_rollouts):
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
            obs, r, done, _ = env.step(action)
            totalr += r
            steps += 1
            if args.render:
                env.render()
            if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
            if steps >= max_steps:
                break
        returns.append(totalr)

    print('returns {}'.format(returns))
    print('mean return {}'.format(np.mean(returns)))
    print('std of return {}'.format(np.std(returns)))
    
    with open(os.path.join('expert_data', save_name + '.json'), 'w') as f:
        json.dump({'returns': returns, 
                    'mean_return': np.mean(returns),
                    'std_return': np.std(returns)}, f)
    
    expert_data = {'observations': np.array(observations),
                    'actions': np.squeeze(np.array(actions), axis=1)}

    with open(os.path.join('expert_data', args.envname + '.pkl'), 'wb') as f:
        pickle.dump(expert_data, f, pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    main()