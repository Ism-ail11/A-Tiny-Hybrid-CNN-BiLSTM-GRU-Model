import torch
import torch.nn as nn
import torch.nn.functional as F

class TemperatureScaler(nn.Module):
    def __init__(self, init_temp: float = 1.0):
        super().__init__()
        self.log_temp = nn.Parameter(torch.log(torch.tensor([init_temp], dtype=torch.float32)))

    def forward(self, logits):
        t = torch.exp(self.log_temp).clamp(1e-3, 100.0)
        return logits / t

    @property
    def temperature(self) -> float:
        return float(torch.exp(self.log_temp).item())

def fit_temperature(model, loader, device='cpu', max_iter=200):
    """Fits a single temperature on validation set to minimize NLL."""
    scaler = TemperatureScaler(1.0).to(device)
    model.eval()
    nll = nn.CrossEntropyLoss()

    logits_list = []
    y_list = []
    with torch.no_grad():
        for x, y, _ in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            logits_list.append(logits.detach().cpu())
            y_list.append(y.detach().cpu())
    logits = torch.cat(logits_list, dim=0).to(device)
    y = torch.cat(y_list, dim=0).to(device)

    optimizer = torch.optim.LBFGS(scaler.parameters(), lr=0.1, max_iter=max_iter)

    def closure():
        optimizer.zero_grad(set_to_none=True)
        loss = nll(scaler(logits), y)
        loss.backward()
        return loss

    optimizer.step(closure)
    return scaler
