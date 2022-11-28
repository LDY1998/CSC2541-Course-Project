import torch
import numpy as np


class HopperStateEncoder:

    def __init__(self, confounded, drop_dims, mean, std, factor_model=None):
        self.confounded = confounded
        self.drop_dims = drop_dims
        self.mean = mean
        self.std = std
        self.factor_model = factor_model

    def batch(self, batch):
        # For imitation learning
        assert batch.states.shape[1] >= 2
        state = batch.states[:, -1, :]

        # Observed confounder
        if self.confounded:
            prev_action = batch.actions[:, -2]
        else:
            prev_action = -2 * torch.rand((state.shape[0], 3), device=state.device) + 1
        state = torch.cat([state.float(), prev_action.float()], dim=-1)

        # Unobserved confounder
        if len(self.drop_dims):
            kept_dims = [i for i in range(state.size(-1)) if i not in self.drop_dims]
            state = state[:, kept_dims]

        # Standardize
        mean = torch.FloatTensor(self.mean, device=state.device)
        std = torch.FloatTensor(self.std, device=state.device)
        state = (state - mean) / std

        # Deconfounder
        if batch.deconfounders is not None:
            state = torch.cat([state, batch.deconfounders[:, -1, :]], dim=-1)

        return state

    def step(self, state, trajectory):
        # For running in environment
        assert state.ndim == 1

        # Observed confounder
        if trajectory and self.confounded:
            prev_action = trajectory.actions[-1]
        else:
            prev_action = -2 * np.random.rand(3) + 1
        state = np.concatenate([state, prev_action])

        # Unobserved confounder
        if len(self.drop_dims):
            kept_dims = [i for i in range(state.shape[-1]) if i not in self.drop_dims]
            state = state[kept_dims]

        # Standardize
        state = (state - self.mean) / self.std

        # Deconfounder
        if self.factor_model is not None:
            state = np.concatenate([state, self.factor_model.predict(state.reshape(1, -1)).reshape(-1)])

        return state
