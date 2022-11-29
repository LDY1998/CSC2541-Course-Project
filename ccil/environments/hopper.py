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

        # Unobserved confounder
        if len(self.drop_dims):
            kept_dims = [i for i in range(state.size(-1)) if i not in self.drop_dims]
            state = state[:, kept_dims]

        # Observed confounder
        mean = torch.FloatTensor(self.mean, device=state.device)
        std = torch.FloatTensor(self.std, device=state.device)
        if self.confounded:
            prev_action = batch.actions[:, -2]
            state = torch.cat([state.float(), prev_action.float()], dim=-1)
            state = (state - mean) / std
        else:
            state = (state.float() - mean) / std
            prev_action = -2 * torch.rand((state.shape[0], 3), device=state.device) + 1
            state = torch.cat([state.float(), prev_action.float()], dim=-1)

        # Deconfounder
        if batch.deconfounders is not None:
            state = torch.cat([state, batch.deconfounders[:, -1, :]], dim=-1)

        return state

    def step(self, state, trajectory):
        # For running in environment
        assert state.ndim == 1

        # Unobserved confounder
        if len(self.drop_dims):
            kept_dims = [i for i in range(state.shape[-1]) if i not in self.drop_dims]
            state = state[kept_dims]

        # Observed confounder
        if self.confounded:
            if trajectory is None:
                prev_action = -2 * np.random.rand(3) + 1
            else:
                prev_action = trajectory.actions[-1]
            state = np.concatenate([state, prev_action])
            state = (state - self.mean) / self.std
        else:
            state = (state - self.mean) / self.std
            prev_action = -2 * np.random.rand(3) + 1
            state = np.concatenate([state, prev_action])

        # Deconfounder
        if self.factor_model is not None:
            if not self.confounded:
                state = np.concatenate([state, self.factor_model.predict(state[:-3].reshape(1, -1)).reshape(-1)])
            else:
                state = np.concatenate([state, self.factor_model.predict(state.reshape(1, -1)).reshape(-1)])

        return state
