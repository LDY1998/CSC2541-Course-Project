import pickle
import argparse
from datetime import datetime
from functools import partial

import gym
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from ignite.engine import Engine, Events
from ignite.metrics import MeanAbsoluteError, Loss
import numpy as np
from pytorch_lightning import seed_everything
seed_everything(42)

# from deconfounder.deconfounder import factor_model as load_factor_model
from deconfounder.vae import factor_model as load_factor_model
from ccil.environments.hopper import HopperStateEncoder
from ccil.utils.data import random_split, batch_cat, DataLoaderRepeater, Trajectory, TransitionDataset
from ccil.utils.models import SimplePolicy, MLP, UniformMaskPolicy
from ccil.utils.policy_runner import PolicyRunner, RandomMaskPolicyAgent, FixedMaskPolicyAgent
from ccil.utils.utils import random_mask_from_state, data_root_path, mask_idx_to_mask


def train_step(engine, batch, state_encoder, policy_model, optimizer, criterion, device):
    x, y = state_encoder.batch(batch), batch.labels()
    x, y = x.to(device), y.to(device)

    mask = random_mask_from_state(x)
    output = policy_model.forward(x, mask)
    loss = criterion(output, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss


def inference_step(engine, batch, state_encoder, policy_model, device):
    x, y = state_encoder.batch(batch), batch.labels()
    x, y = x.to(device), y.to(device)
    mask = random_mask_from_state(x)
    output = policy_model.forward(x, mask)
    return output, y


def print_metrics(engine, trainer, evaluator_name):
    print(
        f"Epoch: {trainer.state.epoch:> 3} {evaluator_name.title(): <5} "
        f"loss={engine.state.metrics['loss']:.4f} "
        f"mae={engine.state.metrics['mae']:.4f}")


def run_simple(policy_model, state_encoder):
    """
    Run the policy in environment.
    """
    env = gym.make("Hopper-v2")
    agent = RandomMaskPolicyAgent(policy_model)
    runner = PolicyRunner(env, agent, state_encoder)
    trajectories = runner.run_num_episodes(20)
    print(f'Mean reward: {Trajectory.reward_sum_mean(trajectories)}')


def run_uniform(policy_model, state_encoder):
    """
    Run all 8 policies in environment.
    """
    env = gym.make("Hopper-v2")
    for mask_idx in range(8):
        agent = FixedMaskPolicyAgent(policy_model, mask_idx_to_mask(3, mask_idx))
        runner = PolicyRunner(env, agent, state_encoder)
        trajectories = runner.run_num_episodes(20)
        mask = mask_idx_to_mask(3, mask_idx).tolist()
        print(f'Mean reward mask {mask}: {Trajectory.reward_sum_mean(trajectories)}')


def load_dataset(confounded, drop_dims, latent_dim):
    # Load deconfounders
    obj = load_factor_model(confounded, drop_dims, latent_dim)
    factor_model = obj['regr']
    mean, std = obj['npz_dic']['mean'], obj['npz_dic']['std']
    deconfounders = obj['npz_dic']['zs']

    # Load expert data
    # Keys: ['observations', 'actions', 'timesteps', 'trajectories']
    expert_path = './expert_data/Hopper-v2.pkl'
    with open(expert_path, 'rb') as fin:
        obj = pickle.load(fin)
    states, actions = obj['observations'], obj['actions']
    traj_ids = obj['trajectories']

    trajectories = []
    for traj_id in np.unique(traj_ids):
        traj_indices = traj_ids == traj_id
        num_steps = np.count_nonzero(traj_indices)
        rewards = np.zeros(num_steps)
        pixels = np.zeros(num_steps)

        trajectory = Trajectory(states[traj_indices], actions[traj_indices], rewards, pixels)
        trajectory.finished()
        trajectories.append(trajectory)

    dataset = TransitionDataset.from_trajectories(
        trajectories, deconfounders, stack_size=2, expert_trajectories=True
    )

    # Build state encoder
    state_encoder = HopperStateEncoder(confounded, drop_dims, mean, std, factor_model)
    return dataset, state_encoder


def imitate(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset, state_encoder = load_dataset(args.confounded, args.drop_dims, args.latent_dim)
    train_dataset, test_dataset = random_split(dataset, [args.num_samples, args.num_samples], args.data_seed)

    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=batch_cat),
        'test': DataLoader(test_dataset, batch_size=64, shuffle=True, collate_fn=batch_cat),
    }
    # So that 1 train epoch has fixed number of samples (500 batches) regardless of dataset size
    dataloaders['train_repeated'] = DataLoaderRepeater(dataloaders['train'], 500)

    if args.network == 'simple':
        input_dim = state_encoder.step(dataset[0].states[0, -1].numpy(), None).shape[-1]
        policy_model = SimplePolicy(MLP([input_dim, 50, 50, 3])).to(device)
        max_epochs = 10
    elif args.network == 'uniform':
        input_dim = 2 * state_encoder.step(dataset[0].states[0, -1].numpy(), None).shape[-1]
        policy_model = UniformMaskPolicy(MLP([input_dim, 50, 50, 50, 3])).to(device)
        max_epochs = 20
    else:
        raise ValueError()

    optimizer = torch.optim.Adam(policy_model.parameters())

    def criterion(x, y):
        return F.mse_loss(x, y)

    metrics = {
        'loss': Loss(F.mse_loss),
        'mae': MeanAbsoluteError(),
    }

    trainer = Engine(partial(
        train_step, state_encoder=state_encoder, policy_model=policy_model,
        optimizer=optimizer, criterion=criterion, device=device
    ))
    evaluators = {
        name: Engine(partial(
            inference_step, state_encoder=state_encoder, policy_model=policy_model, device=device))
        for name in ['train', 'test']}
    for evaluator_name, evaluator in evaluators.items():
        for name, metric in metrics.items():
            metric.attach(evaluator, name)
        evaluator.add_event_handler(Events.COMPLETED, print_metrics, evaluator_name=evaluator_name, trainer=trainer)

    @trainer.on(Events.EPOCH_COMPLETED)
    def run_eval(_trainer):
        for name, evaluator in evaluators.items():
            evaluator.run(dataloaders[name])

    trainer.run(dataloaders['train_repeated'], max_epochs=max_epochs)
    print("Trained")

    # Run policies in environment
    # run_fn = dict(simple=run_simple, uniform=run_uniform)[args.network]
    # run_fn(policy_model, state_encoder)
    if args.network == 'simple':
        run_simple(policy_model, state_encoder)

    if args.save:
        name = args.name or f"{args.input_mode}_{args.network}_{datetime.now():%Y%m%d-%H%M%S}"
        save_dir = data_root_path / 'policies'
        save_dir.mkdir(parents=True, exist_ok=True)
        path = save_dir / f"{name}.pkl"
        torch.save(policy_model, path)
        print(f"Policy saved to {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--confounded', action='store_true')
    parser.add_argument('--drop_dims', nargs='+', type=int, default=[])
    parser.add_argument('--latent_dim', type=int, default=-1)

    parser.add_argument('--network', choices=['simple', 'uniform'], required=True)
    parser.add_argument('--data_seed', type=int)
    parser.add_argument('--num_samples', type=int, default=300)
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--name', help="Policy save filename")
    imitate(parser.parse_args())


if __name__ == '__main__':
    main()
