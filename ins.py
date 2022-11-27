import pickle


with open('./expert_data/Trajectories-10_samples-10000_masked-5_confounded_inferred-5.pkl', 'rb') as fin:
    obj = pickle.load(fin)


keys = ['observations', 'actions', 'timesteps', 'trajectories', 'mean', 'std']
for key in obj.keys():
    print(obj[key].shape)
