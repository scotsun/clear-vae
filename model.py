import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


def vae_loss(x_reconstr, x, mu, log_var):
    reconstruction_loss = (
        F.binary_cross_entropy(x_reconstr, x, reduction="none")
        .sum(dim=(1, 2, 3))
        .mean()
    )
    kl_regularization = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
    return reconstruction_loss + kl_regularization


class VAE(nn.Module):
    def __init__(self, z_dim) -> None:
        super().__init__()
        # encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 16),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(16, z_dim)
        self.fc_log_var = nn.Linear(16, z_dim)
        # decoder
        self.dec_fc = nn.Linear(z_dim, 3136)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, 2, 1, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 3, 2, 1, 1),
            nn.Sigmoid(),
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_log_var(h)

    def decode(self, z):
        x = self.dec_fc(z)
        x = x.reshape(-1, 64, 7, 7)
        x = self.decoder(x)
        return x

    def sample(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.sample(mu, log_var)
        return self.decode(z), mu, log_var

    def evaluate(self, dataloader, device, verbose):
        with torch.no_grad():
            num_batch = 0.0
            total_loss = 0.0
            for X_batch, _ in tqdm(dataloader, disable=not verbose):
                X_batch = X_batch.to(device)
                X_reconstr_batch, mu_batch, log_var_batch = self(X_batch)
                loss = vae_loss(X_reconstr_batch, X_batch, mu_batch, log_var_batch)
                num_batch += 1
                total_loss += loss.item()
        return total_loss / num_batch

    def fit(self, train_dataloader, valid_dataloader, num_epoch, lr, device, verbose):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        for epoch in range(num_epoch):
            num_batch = 0.0
            total_loss = 0.0
            self.train()
            with tqdm(
                train_dataloader, unit="batch", mininterval=0, disable=not verbose
            ) as bar:
                bar.set_description(f"Epoch {epoch}")
                for X_batch, _ in bar:
                    optimizer.zero_grad()
                    X_batch = X_batch.to(device)
                    X_reconstr_batch, mu_batch, log_var_batch = self(X_batch)
                    loss = vae_loss(X_reconstr_batch, X_batch, mu_batch, log_var_batch)
                    loss.backward()
                    optimizer.step()
                    # update bar
                    num_batch += 1
                    total_loss += loss.item()
                    bar.set_postfix(loss=float(total_loss / num_batch))
            self.eval()
            valid_loss = self.evaluate(valid_dataloader, device, False)
            print("val_loss={:0.3f}".format(valid_loss))

    def calculate_latent_var(self, dataloader, device):
        Z_mu = []
        for X_batch, _ in tqdm(dataloader):
            X_batch = X_batch.to(device)
            _z_mu, _ = self.encode(X_batch)
            Z_mu.append(_z_mu.detach().cpu().numpy())
        Z_mu = np.row_stack(Z_mu)
        return Z_mu


class MixtureOfExperts(nn.Module):
    def __init__(self, input_dim, num_expert) -> None:
        super().__init__()
        self.gate_mgr = nn.Linear(in_features=input_dim, out_features=num_expert)
        self.experts = nn.Linear(in_features=input_dim, out_features=num_expert)

    def forward(self, x):
        w = F.softmax(self.gate_mgr(x), dim=1)  # TODO: can add sparsity
        expert_scores = F.sigmoid(self.experts(x))
        # return F.sigmoid(torch.sum(w * expert_scores, dim=1))
        return torch.sum(w * expert_scores, dim=1)

    def evaluate(self, dataloader, criterion):
        total_loss = 0.0
        total_sample = 0.0
        with torch.no_grad():
            for X_batch, y_batch in tqdm(dataloader):
                p_batch = self(X_batch)
                total_loss += criterion(p_batch, y_batch).item()
                total_sample += X_batch.shape[0]
        return total_loss / total_sample
