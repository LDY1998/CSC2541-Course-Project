from argparse import Namespace
from glob import glob
import re
import matplotlib.pyplot as plt
import os
from ccil.intervention_policy_execution import intervention_policy_execution
from pathlib import Path


def draw():
    print(os.getcwd())
    data_root_dir = "./data"
    policies_dir = Path(data_root_dir + "/policies")
    policies = [str(os.path.splitext(f)[0]) for f in os.listdir(policies_dir) if re.search(r"(vae|None|fm)_\[4\]", f)]
    print(policies)
    num_iters = 100
    drop_dims = [4]
    temperature = 2
    confounded = True

    plt.xlabel("num iterations")
    plt.ylabel("rewards")
    
    for policy in policies:
        d = {"policy_name": policy, "num_its": num_iters, "drop_dims": drop_dims, "temperature": temperature, 
                "confounded": confounded, "latent_dim": -1, "seed": 24, "deconfounder": policy.split("_")[0]}
        if int(policy.split("_")[2]) > -1:
            d["latent_dim"] = int(policy.split("_")[2])
        ns = Namespace(**d)
        its, rewards = intervention_policy_execution(ns)
        line_label = f"{d['deconfounder']}_drop_{d['drop_dims']}_latent_{d['latent_dim']}"
        plt.plot([i for i in range(its)], smooth(rewards), label=line_label)
            
    plt.legend()
    plt.savefig(f"./graphs/test-{num_iters}-seed-{d['seed']}.png")

def smooth(arr):
    res = []
    for i in range(len(arr)):
        if i == 0 or i == len(arr)-1:
            res.append(arr[i])
        else:
            res.append((arr[i-1]+arr[i]+arr[i+1])/3)

    return res



if __name__ == '__main__':
    draw()