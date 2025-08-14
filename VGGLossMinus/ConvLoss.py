import torch
import torch.nn as nn
import torch.fft
import matplotlib.pyplot as plt
import math

class LRN(nn.Module):
    def forward(self, x):
        return x/x.norm(dim=1).unsqueeze(1)

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

            # Normalize to unit variance
            kernel -= kernel.mean()
            kernel /= kernel.norm()

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
    weights = conv.weight.detach().cpu()
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
    weights = conv.weight.detach().cpu()
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
