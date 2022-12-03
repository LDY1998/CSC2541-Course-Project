import pickle

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


class VAE(nn.Module):

    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, 100), nn.ReLU())
        self.mu_layer = nn.Linear(100, latent_dim)
        self.logvar_layer = nn.Linear(100, latent_dim)
        self.decoder = nn.Sequential(nn.Linear(latent_dim, 100), nn.ReLU(), nn.Linear(100, input_dim))

    def encode(self, x):
        h = self.encoder(x)
        return self.mu_layer(h), self.logvar_layer(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), x, mu, logvar

    def predict(self, x):
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        # print(f"------- device {torch.cuda.is_available()} -------")
        self.to(device)
        x = torch.from_numpy(x).float()
        x = x.to(device)
        mu, _ = self.encode(x)
        return mu.cpu().detach().numpy()

    def loss_function(self, recon_x, x, mu, logvar, kld_weight=1):
        recon_loss = F.mse_loss(recon_x, x)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp(), dim=-1), dim=0)
        loss = recon_loss + kld_weight * kld_loss
        return {'loss': loss, 'recon_loss': recon_loss.detach(), 'kld_loss': -kld_loss.detach()}


def train(num_epochs, model, train_loader, device):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model.to(device)
    model.train()

    for epoch in tqdm(range(num_epochs)):
        for batch in train_loader:
            batch = batch.to(device)
            loss = model.loss_function(*model(batch))

            optimizer.zero_grad()
            loss['loss'].backward()
            optimizer.step()


expert_path = './expert_data/Hopper-v2.pkl'
confounded_path = './expert_data/Trajectories-10_samples-10000_confounded.pkl'


class ExpertDataset(Dataset):

    def __init__(self, confounded, drop_dims):
        super(ExpertDataset, self).__init__()
        pickle_path = confounded_path if confounded else expert_path
        with open(pickle_path, 'rb') as fin:
            states = pickle.load(fin)['observations']
        left_dims = [i for i in range(states.shape[-1]) if i not in drop_dims]
        states = states[:, left_dims]

        self.mean = states.mean(axis=0)
        self.std = states.std(axis=0)
        self.states = (states - self.mean) / self.std

    def __getitem__(self, idx):
        return torch.FloatTensor(self.states[idx])

    def __len__(self):
        return len(self.states)


def factor_model(confounded, drop_dims, latent_dim):
    dataset = ExpertDataset(confounded, drop_dims)
    data = {'regr': None, 'npz_dic': {'mean': dataset.mean, 'std': dataset.std, 'zs': None}}
    if latent_dim == -1:
        return data

    loader = DataLoader(dataset, batch_size=64)
    model = VAE(dataset.states.shape[-1], latent_dim)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    train(10, model, loader, device)
    data['regr'] = model
    data['npz_dic']['zs'] = model.predict(dataset.states)
    return data
