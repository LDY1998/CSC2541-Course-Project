# CSC2541-Course-Project


## Setup

The environment is set up under `python 3.6`, make sure the version is right when create the conda environment.

Make sure to download mujoco through this instruction [install-mujoco](https://github.com/openai/mujoco-py#install-mujoco)

run 
```shell 
conda env create -f environment.yml
```
to install the dependencies

Then we will need to generate the expert data for further use:
```shell
python ./preprocessing/run_expert.py --num_traj=# of trajectories to generate
```

Run the script to preprocess the data:
```shell
python ./preprocessing/data_preprocessing.py
```

After that, we can run either of the 2 commands to run behavior cloning or intervention mask learning:

```shell
python -m ccil.imitate --drop_dims dimentions  --confounded --network uniform --save
```

```shell
python -m ccil.intervention_policy_execution --drop_dims 2 4 --confounded --temperature 2 --num_its 100 --policy_name="[2,4]_-1_uniform_20221202-162924" (policy name)
```
The extra information of hopper dataset can be found at: [https://www.gymlibrary.dev/environments/mujoco/hopper/#hopper](Hopper)
