import torch
import torch.nn as nn
import itertools
import math
import matplotlib.pyplot as plt

def visualize_kernel(kernel, edge_size):
    plt.subplots(edge_size, edge_size)
    for i in range(edge_size**2):
        plt.subplot(edge_size, edge_size, i+1)
        plt.imshow(kernel[i])


def test_conv_basis(basis: torch.Tensor, tol: float = 1e-6, verbose: bool = True):
    """
    Universal tester for convolutional bases.

    Args:
        basis: torch.Tensor of shape (num_filters, H, W)
        tol: tolerance for floating-point checks
        verbose: whether to print detailed results

    Returns:
        dict with test results
    """
    num_filters, H, W = basis.shape
    expected = H * W

    results = {}

    if verbose:
        print(f"Testing {num_filters} filters for {H}x{W} kernels")
        print(f"Expected {expected} independent filters")

    # Flatten basis into vectors
    flat = basis.view(num_filters, -1)

    # --- Completeness check ---
    results["completeness"] = (num_filters == expected)
    if verbose:
        print("✔ Correct number of filters" if results["completeness"] 
              else f"❌ Wrong number of filters ({num_filters} vs {expected})")

    # --- Orthonormality check ---
    G = flat @ flat.T   # Gram matrix

    # Check norms ~ 1
    diag = torch.diag(G)
    results["unit_norms"] = torch.allclose(diag, torch.ones_like(diag), atol=tol)
    if verbose:
        print("✔ Unit norms" if results["unit_norms"] else "❌ Norms not 1")

    # Check orthogonality (off-diagonal ~ 0)
    off_diag = G - torch.diag(diag)
    results["orthogonal"] = torch.allclose(off_diag, torch.zeros_like(off_diag), atol=tol)
    if verbose:
        print("✔ Orthogonal" if results["orthogonal"] else "❌ Not orthogonal")

    # --- Span check ---
    # Rank of flattened basis should equal expected dimension
    rank = torch.linalg.matrix_rank(flat)
    results["span"] = (rank == expected)
    if verbose:
        print("✔ Spans full space" if results["span"] else f"❌ Rank deficient (rank={rank})")

    # --- Overall ---
    results["valid_basis"] = all(results.values())
    if verbose:
        if results["valid_basis"]:
            print("🎉 Valid orthonormal basis!")
        else:
            print("⚠ Basis failed one or more checks")

    return results

def unique_fourier_basis_2d(size: int, center_basis: bool=False) -> torch.Tensor:
    """
    Generate a unique, orthogonal real Fourier basis (cos/sin) for a size x size kernel.
    Avoids duplicates from conjugate symmetry.
    """
    n = size
    xs = torch.arange(n).float()
    ys = torch.arange(n).float()
    grid_x, grid_y = torch.meshgrid(xs, ys, indexing="ij")

    basis = []

    for u in range(n):
        for v in range(n):
            angle = 2 * math.pi * (u * grid_x / n + v * grid_y / n)

            cos_f = torch.cos(angle)
            sin_f = torch.sin(angle)

            # normalize
            cos_f = cos_f / cos_f.norm()
            sin_norm = sin_f.norm()

            # add cosine (always unique by construction)
            basis.append(cos_f)

            # add sine only if it's not (almost) zero
            if sin_norm > 1e-6:
                sin_f = sin_f / sin_norm
                basis.append(sin_f)

    # stack and deduplicate (within numerical tolerance)
    B = torch.stack(basis)  # (num_candidates, n, n)
    flat = B.view(B.shape[0], -1)

    keep = []
    final = []
    for i in range(flat.shape[0]):
        v = flat[i]
        # check if already in span of chosen ones
        if not keep:
            keep.append(v)
            final.append(B[i])
        else:
            proj = sum((v @ k) * k for k in keep)
            residual = v - proj
            if residual.norm() > 1e-6:  # independent
                keep.append(v / v.norm())
                final.append(B[i])

    if center_basis:
        n = size
        # center impulse
        center = torch.zeros(n, n)
        center[n//2, n//2] = 1.0
        final[0] = center

    return torch.stack(final)  # (num_filters, n, n)



def make_conv(in_channels: int, size: int, function: callable) -> nn.Conv2d:
    basis = function(size)  # (out_channels, size, size)
    out_channels = basis.shape[0]*in_channels

    conv = nn.Conv2d(in_channels, out_channels, kernel_size=size, bias=False, groups=in_channels)
    # Repeat basis for each input channel
    weight = basis.repeat(in_channels, 1, 1).unsqueeze(1)

    conv.weight.data = weight

    # Freeze if you don’t want training
    for p in conv.parameters():
        p.requires_grad = False

    return conv

class KernelLoss(nn.Module):
    def __init__(self, in_channels: int, kernel_size: int, function: callable, identity_weight: float, loss_func):
        super().__init__()
        self.conv =  make_conv(in_channels, kernel_size, function)
        self.identity_weight = identity_weight
        self.loss_func = loss_func()
        self.in_channels = in_channels

    def forward(self, x1, x2):
        x1 = self.conv(x1)
        x2 = self.conv(x2)
        loss = 0
        out_channels = x2.shape[1]
        for i in range(out_channels):
            if (i % 9)==0: 
                loss += self.identity_weight * self.loss_func(x1[:, i, :, : ], x2[:, i, :, : ])
            else:
                loss += (1-self.identity_weight) * self.loss_func(x1[:, i, :, : ], x2[:, i, :, : ])
        return loss/self.in_channels
