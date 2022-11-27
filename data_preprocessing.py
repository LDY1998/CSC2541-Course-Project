from copy import copy
import numpy as np
import pickle
import os

MASK_IDX = 5
FILE_PATH = "expert_data/Hopper-v2.pkl"

def main(filename):
    with open(filename, 'rb') as f:
        data = pickle.loads(f.read())

    num_rollouts = data['trajectories'][-1]+1
    n = data["observations"].shape[0]
    # confounded data
    c_data = copy(data)
    m_data = copy(data)
    actions = c_data["actions"]
    timesteps = c_data["timesteps"]
    actions[1:-1] = actions[0:-2]
    actions[np.where(timesteps == 0)] = np.zeros(3)
    c_obs = np.concatenate((c_data["observations"], actions), axis=1)
    c_data["observations"] = c_obs
    c_file_name = f"Trajectories-{num_rollouts}_samples-{n}_confounded"
    with open(os.path.join('expert_data', c_file_name + '.pkl'), 'wb') as f:
        pickle.dump(c_data, f, pickle.HIGHEST_PROTOCOL)


    # produce mask data
    masked_obs = np.delete(m_data["observations"], MASK_IDX, axis=1)
    m_data["observations"] = masked_obs
    m_file_name = f"Trajectories-{num_rollouts}_samples-{n}_masked-{MASK_IDX}"


    with open(os.path.join('expert_data', m_file_name + '.pkl'), 'wb') as f:
        pickle.dump(m_data, f, pickle.HIGHEST_PROTOCOL)

    data["observations"] = masked_obs
    data["observations"] = np.concatenate((data["observations"], actions), axis=1)
 
    cm_f = f"Trajectories-{num_rollouts}_samples-{n}_masked-{MASK_IDX}_confounded"
    with open(os.path.join('expert_data', cm_f + '.pkl'), 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    main(FILE_PATH)
    # with open("expert_data/Trajectories-10_samples-10000_masked-5.pkl", 'rb') as f:
    #     data = pickle.loads(f.read())
    #     print(data)