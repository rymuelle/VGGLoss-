import torch.nn as nn
import torch

# Refernece goes here
class PeriodicLinearUnit(nn.Module):
    def __init__(self, num_parameters=1, init_alpha=1.0, init_beta=1.0, init_rho_alpha=5.0, init_rho_beta=0.15):
        super().__init__()
        self.alpha = nn.Parameter(torch.full((num_parameters,), init_alpha)).to(device)
        self.beta = nn.Parameter(torch.full((num_parameters,), init_beta)).to(device)
        self.rho_alpha = nn.Parameter(torch.full((num_parameters,), init_rho_alpha)).to(device)
        self.rho_beta = nn.Parameter(torch.full((num_parameters,), init_rho_beta)).to(device)
    
    def forward(self, x):
        # repulsive reparameterization / asymptotic regularization
        alpha_eff = self.alpha + self.rho_alpha / self.alpha
        beta_eff = self.beta + self.rho_beta / self.beta
        return x + (beta_eff / (1.0 + torch.abs(beta_eff))) * torch.sin(torch.abs(alpha_eff) * x)
    

# Reference goes here
class IsotropicAct(nn.Module):
    def __init__(self, act_func=nn.functional.tanh):
        super(IsotropicAct, self).__init__()

        # Since tanh approximates the identity around the origin, then a small ball is just treated as the identity map this ensures computational stability. This is because there is a coordinate artefact about the centre, which shouldn't but does meaningfully affect computation due to precision. Therefore, avoid this with this threshold.
        self.epsilon = 1e-2
        self.act_func = act_func

    def forward(self, x: torch.Tensor, dims:tuple[int, ...]=(-1,))-> torch.Tensor:
        # Calculate the vector magnitude
        magnitude = torch.linalg.norm(x, dim=dims, keepdim=True)
        # A small ball approximates the identity map for more stable computation
        identity_mask = magnitude <= self.epsilon
        # Calculate the required unit-vector
        unit_vector = x / magnitude.clamp(min=self.epsilon)
        # Return final computation, identity map for when the magnitude is small, otherwise Isotropic Tanh
        return torch.where(identity_mask, x, self.act_func(magnitude) * unit_vector)
    

## Sugar reference

# BSiLU activation function
def bsilu(x: torch.Tensor) -> torch.Tensor:
    return (x + 1.67) * torch.sigmoid(x) - 0.835


class SurrogateAct(nn.Module):
    def __init__(self, forward_act=nn.functional.relu, backwards_act=bsilu):
        super().__init__()
        self.forward_act = forward_act
        self.backward_act = backwards_act

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gx = self.backward_act(x)
        return gx - gx.detach() + self.forward_act(x).detach()
        

class Sine(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(x)

class Cosine(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cos(x) 
    
class BSiLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return bsilu(x)
    

activations = {
    'ReLU_BSiLU': (SurrogateAct, 1),
    'identity': (nn.Identity, 1),
    'relu': (nn.ReLU, 1),
    'gelu': (nn.GELU, 1),
    'silu': (nn.SiLU, 1),
    'leaky': (nn.LeakyReLU, 1),
    'IsotropicTanh': (IsotropicAct, 1),
    'Mish': (nn.Mish, 1),
    'plu': (PeriodicLinearUnit, 13),
    'tanh': (nn.Tanh, 1),
    'BSiLU': (BSiLU,1),
    'Sine': (Sine, 1),
    'Cosine': (Cosine, 1)
    
}