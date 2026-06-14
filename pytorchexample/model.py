"""CNN model for FashionMNIST federated learning."""

import numpy as np
import torch
import torch.nn as nn


class CustomFashionModel(nn.Module):
    """Simple CNN for FashionMNIST (28x28 grayscale, 10 classes)."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),  # 28x28 -> 28x28
            nn.ReLU(),
            nn.MaxPool2d(2),                  # 28x28 -> 14x14
            nn.Conv2d(32, 64, 3, padding=1),  # 14x14 -> 14x14
            nn.ReLU(),
            nn.MaxPool2d(2),                  # 14x14 -> 7x7
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.net(x)

    def train_epoch(self, train_loader, criterion, optimizer, device):
        """Train for one full epoch, return (avg_loss, accuracy)."""
        self.train()
        total_loss, correct, total = 0.0, 0, 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = self(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)

        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total
        return avg_loss, accuracy

    def test_epoch(self, test_loader, criterion, device):
        """Evaluate on the given loader, return (avg_loss, accuracy)."""
        self.eval()
        total_loss, correct, total = 0.0, 0, 0

        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                out = self(x)
                total_loss += criterion(out, y).item()
                correct += (out.argmax(1) == y).sum().item()
                total += y.size(0)

        avg_loss = total_loss / len(test_loader)
        accuracy = correct / total
        return avg_loss, accuracy

    def get_model_parameters(self):
        """Return model weights as a list of numpy arrays."""
        return [val.cpu().detach().numpy() for val in self.state_dict().values()]

    def set_model_parameters(self, params):
        """Load a list of numpy arrays back into the model's state dict."""
        keys = list(self.state_dict().keys())
        state_dict = {k: torch.tensor(p) for k, p in zip(keys, params)}
        self.load_state_dict(state_dict, strict=True)

    def train_one_step(self, train_loader, criterion, optimizer, device):
        """Take exactly one batch, do a forward+backward pass, return (loss, acc).

        Used for FedSGD-style training where clients do a single gradient step.
        """
        self.train()
        x, y = next(iter(train_loader))  # grab the first batch only
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = self(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        acc = (out.argmax(1) == y).float().mean().item()
        return loss.item(), acc
