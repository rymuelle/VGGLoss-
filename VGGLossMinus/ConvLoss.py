import torch
import torch.nn as nn
import torch.fft
import matplotlib.pyplot as plt
import math
import copy

class LRN(nn.Module):
    def forward(self, x):
        return x/(x.norm(dim=1).unsqueeze(1)+1e-3)

class ConvLoss(nn.Module):
    def __init__(self, in_chan=3, out_chan=150, kernel_size=3, 
                 norm = nn.Identity, 
                 loss_func = nn.functional.mse_loss,
                 activation = nn.Identity,
                 **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_chan, out_chan, kernel_size, **kwargs)
        self.norm = norm()
        self.loss_func = loss_func
        self.activation = activation()
        self.conv.weight.requires_grad_(False)

    def run_conv(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x
    def forward(self, x1, x2):
        x1 = self.run_conv(x1)
        x2 = self.run_conv(x2)
        return self.loss_func(x1, x2)

def init_randn(module):
    if isinstance(module, nn.Conv2d):
        with torch.no_grad():
            module.weight.copy_(torch.randn_like(module.weight))
            if module.bias is not None:
                module.bias.zero_()

def init_conv_fourier(module):
    if isinstance(module, nn.Conv2d):

        weight = module.weight
        out_channels, in_channels, kh, kw = weight.shape

        # Random complex coefficients (Fourier domain)
        # Here we use real & imaginary parts as N(0, 1)
        freq_real = torch.randn(out_channels, in_channels, kh, kw)
        freq_imag = torch.randn(out_channels, in_channels, kh, kw)
        freq = torch.complex(freq_real, freq_imag)

        # Inverse FFT to spatial domain
        spatial = torch.fft.ifft2(freq).real

        # Normalize (optional) to control variance
        spatial = spatial / spatial.std()

        # Assign in-place so it stays a Parameter
        with torch.no_grad():
            weight.copy_(spatial)

            if module.bias is not None:
                module.bias.zero_()


def init_conv_fourier_smooth(module, alpha=1.0):
    """Initialize Conv2d weights with low-frequency bias in Fourier space."""
    if isinstance(module, nn.Conv2d):
        weight = module.weight
        out_channels, in_channels, kh, kw = weight.shape

        # Frequency grid
        fy = torch.fft.fftfreq(kh)[:, None]  # shape (kh, 1)
        fx = torch.fft.fftfreq(kw)[None, :]  # shape (1, kw)
        freq_radius = torch.sqrt(fx**2 + fy**2)  # normalized radius in [0, 0.5]

        # Avoid divide-by-zero at DC
        freq_radius[0,0] = 1e-6  

        # 1/f^alpha amplitude falloff
        amplitude = 1.0 / (freq_radius ** alpha)

        # Random complex coefficients
        freq_real = torch.randn(out_channels, in_channels, kh, kw)
        freq_imag = torch.randn(out_channels, in_channels, kh, kw)
        freq = torch.complex(freq_real, freq_imag)

        # Apply frequency shaping
        freq = freq * amplitude  # broadcast over channels

        # Inverse FFT to spatial domain
        spatial = torch.fft.ifft2(freq).real

        # Normalize variance
        spatial = spatial / spatial.std()

        with torch.no_grad():
            weight.copy_(spatial)
            if module.bias is not None:
                module.bias.zero_()


import random
def init_conv_fourier_mode(conv):
    """
    Initialize Conv2d kernels with an orthogonal 2D discrete Fourier basis.
    Each kernel gets a unique (fx, fy) frequency pair.
    """
    C_out, C_in, H, W = conv.weight.shape
    total_filters = C_out * C_in

    # List all possible frequency pairs for the given kernel size
    def rand_freq(max):
        return random.random()*max
    freqs = [(rand_freq(H), rand_freq(W)) for f in range(total_filters)]

    with torch.no_grad():
        for idx in range(total_filters):
            fx, fy = freqs[idx]
            # Frequency coordinates
            u = torch.arange(H).reshape(H, 1)
            v = torch.arange(W).reshape(1, W)

            # Complex exponential in spatial domain (real part is cosine pattern)
            kernel = torch.cos(2 * math.pi * (fx * u / H + fy * v / W))

            kernel -= kernel.mean()
            kernel /= kernel.std()

            # Assign to correct (out, in) location
            out_ch = idx // C_in
            in_ch = idx % C_in
            conv.weight[out_ch, in_ch] = kernel

    # Optional bias init
    if conv.bias is not None:
        nn.init.zeros_(conv.bias)


def visualize_kernels_color(conv, num_cols=8, cmap='gray'):
    """
    Visualize Conv2d kernels in spatial domain.
    Assumes shape (C_out, C_in, H, W).
    Displays only the first input channel for clarity.
    """
    weights =  copy.deepcopy(conv.weight.detach().cpu())
    C_out, C_in, H, W = weights.shape
    fig, axes = plt.subplots(
        math.ceil(C_out / num_cols), num_cols,
        figsize=(num_cols, math.ceil(C_out / num_cols))
    )
    axes = axes.flatten()
    for i in range(C_out):
        kernel = weights[i]  # First input channel
        kernel -= kernel.min()
        kernel *= 1/kernel.max()

        axes[i].imshow(kernel.permute(1, 2, 0))
        axes[i].axis('off')
    plt.tight_layout()
    plt.show()
    
def visualize_kernels(conv, num_cols=8, cmap='gray'):
    """
    Visualize Conv2d kernels in spatial domain.
    Assumes shape (C_out, C_in, H, W).
    Displays only the first input channel for clarity.
    """
    weights = copy.deepcopy(conv.weight.detach().cpu())
    C_out, C_in, H, W = weights.shape
    fig, axes = plt.subplots(
        math.ceil(C_out / num_cols), num_cols,
        figsize=(num_cols, math.ceil(C_out / num_cols))
    )
    axes = axes.flatten()
    for i in range(C_out):
        kernel = weights[i, 0]  # First input channel
        axes[i].imshow(kernel, cmap=cmap)
        axes[i].axis('off')
    plt.tight_layout()
    plt.show()
    
    C_out, C_in, H, W = weights.shape
    fig, axes = plt.subplots(
        math.ceil(C_out / num_cols), num_cols,
        figsize=(num_cols, math.ceil(C_out / num_cols))
    )
    axes = axes.flatten()
    for i in range(C_out):
        kernel = weights[i, 1]  # First input channel
        axes[i].imshow(kernel, cmap=cmap)
        axes[i].axis('off')
    plt.tight_layout()
    plt.show()

    C_out, C_in, H, W = weights.shape
    fig, axes = plt.subplots(
        math.ceil(C_out / num_cols), num_cols,
        figsize=(num_cols, math.ceil(C_out / num_cols))
    )
    axes = axes.flatten()
    for i in range(C_out):
        kernel = weights[i, 2]  # First input channel
        axes[i].imshow(kernel, cmap=cmap)
        axes[i].axis('off')
    plt.tight_layout()
    plt.show()


import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

class MultiScaleConvLoss(nn.Module):
    def __init__(self, 
                 in_chan=3, 
                 scales=(3, 5, 9),    # kernel sizes for each scale
                 out_chan=64,         # feature channels per scale
                 norm=nn.Identity, 
                 activation=nn.Identity,
                 loss_func=partial(F.mse_loss, reduction='mean'),
                 freeze_conv=False,
                 **kwargs):
        """
        Multi-scale convolutional loss.
        Args:
            scales: tuple of kernel sizes
            out_chan: number of output channels per scale
            freeze_conv: if True, conv weights are frozen (fixed random)
            norm, activation: modules or callables
            loss_func: callable loss function taking (pred, target)
        """
        super().__init__()
        
        self.branches = nn.ModuleList()
        for k in scales:
            branch = nn.Sequential(
                nn.Conv2d(in_chan, out_chan, kernel_size=k, padding=k//2, **kwargs),
                norm(out_chan) if norm != nn.Identity else norm(),
                activation() if activation != nn.Identity else activation()
            )
            if freeze_conv:
                for p in branch[0].parameters():
                    p.requires_grad_(False)
            self.branches.append(branch)
        
        self.loss_func = loss_func

    def forward(self, x1, x2):
        total_loss = 0.0
        for branch in self.branches:
            f1 = branch(x1)
            f2 = branch(x2)
            total_loss += self.loss_func(f1, f2)
        return total_loss / len(self.branches)



import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import math

def make_band_kernel(size, band, bandwidth, device):
    """
    Create a 2D band-pass kernel in the Fourier domain and return its spatial form.
    band: center frequency in [0, 0.5] (0 = DC, 0.5 = Nyquist)
    bandwidth: fractional width of the band
    """
    # freq coords
    fy = torch.fft.fftfreq(size, d=1.0, device=device)[:, None]
    fx = torch.fft.fftfreq(size, d=1.0, device=device)[None, :]
    r = torch.sqrt(fx**2 + fy**2)

    # Gaussian-shaped band mask
    sigma = bandwidth / 2.355  # convert bandwidth to Gaussian sigma
    mask = torch.exp(-0.5 * ((r - band) / sigma)**2)

    # Inverse FFT to get spatial kernel
    kernel = torch.fft.ifft2(mask).real
    kernel = torch.fft.fftshift(kernel)  # center in spatial domain
    kernel = kernel / kernel.abs().sum() # normalize energy
    return kernel.float()

class FourierBandConvLoss(nn.Module):
    def __init__(self,
                 in_chan=3,
                 out_chan=32,
                 bands=((0.0, 0.15), (0.15, 0.3), (0.3, 0.5)),
                 kernel_size=33,
                 loss_func=partial(F.mse_loss, reduction='mean'),
                 freeze=True):
        """
        Multi-band Fourier convolution loss.
        bands: list of (center_freq, bandwidth) in normalized frequency units.
        """
        super().__init__()
        self.loss_func = loss_func
        self.branches = nn.ModuleList()

        for center, bw in bands:
            k = make_band_kernel(kernel_size, center, bw, device='cpu')
            k = k[None, None]  # (1,1,H,W)
            # Duplicate kernel for all input channels
            k = k.repeat(out_chan, in_chan, 1, 1)
            conv = nn.Conv2d(in_chan, out_chan, kernel_size,
                             padding=kernel_size//2, bias=False)
            with torch.no_grad():
                conv.weight.copy_(k)
            if freeze:
                for p in conv.parameters():
                    p.requires_grad_(False)
            self.branches.append(conv)

    def forward(self, x1, x2):
        total_loss = 0.0
        for branch in self.branches:
            f1 = branch(x1)
            f2 = branch(x2)
            total_loss += self.loss_func(f1, f2)
        return total_loss / len(self.branches)
