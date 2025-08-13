import torch
import torch.nn.functional as F
import numpy as np
from skimage import color, filters
from skimage.metrics import structural_similarity as ssim
from scipy.fft import fft2, fftshift
from scipy.stats import wasserstein_distance
import lpips
import cv2
import pandas as pd
from piq import gmsd

# -------------------------
# Helpers
# -------------------------

def to_numpy(img):
    """Converts torch.Tensor [C,H,W] in [0,1] to np.uint8 [H,W,C]"""
    if torch.is_tensor(img):
        img = img.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    return (img * 255).astype(np.uint8)

def to_float_np(img):
    """Converts torch.Tensor [C,H,W] in [0,1] to np.float32 [H,W,C]"""
    if torch.is_tensor(img):
        img = img.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    return img.astype(np.float32)

def delta_e2000(img1, img2):
    """CIEDE2000 color difference"""
    lab1 = color.rgb2lab(img1)
    lab2 = color.rgb2lab(img2)
    return np.mean(color.deltaE_ciede2000(lab1, lab2))

def hue_error(img1, img2):
    hsv1 = color.rgb2hsv(img1)
    hsv2 = color.rgb2hsv(img2)
    return np.mean(np.abs(hsv1[...,0] - hsv2[...,0]))

def high_freq_energy_ratio(img1, img2):
    """Fourier high-frequency energy ratio"""
    def hf_energy(img):
        gray = color.rgb2gray(img)
        f = np.abs(fftshift(fft2(gray)))
        h, w = f.shape
        mask = np.ones_like(f)
        mask[h//4:3*h//4, w//4:3*w//4] = 0
        return np.sum(f * mask) / np.sum(f)
    return hf_energy(img1) / hf_energy(img2)

def local_contrast(img):
    gray = color.rgb2gray(img)
    return np.std(gray)

def histogram_emd(img1, img2):
    """Earth Mover’s Distance between histograms"""
    hist1, _ = np.histogram(img1.ravel(), bins=256, range=(0, 255), density=True)
    hist2, _ = np.histogram(img2.ravel(), bins=256, range=(0, 255), density=True)
    return wasserstein_distance(hist1, hist2)

def niqe_score(img):
    """No-reference NIQE metric (requires OpenCV contrib or skimage-niqe)"""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    try:
        from skimage.metrics import niqe
        return niqe(gray)
    except ImportError:
        return np.nan

# -------------------------
# Main evaluation function
# -------------------------

def evaluate_image_pairs(image_pairs, device):
    """
    image_pairs: list of (pred, gt) images, each [C,H,W] tensor in [0,1]
    Returns: Pandas DataFrame with metrics
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Preload LPIPS
    lpips_fn = lpips.LPIPS(net='alex').to(device)

    results = []
    for pred, gt in image_pairs:
        pred_np = to_numpy(pred)
        gt_np = to_numpy(gt)
        pred_f = to_float_np(pred)
        gt_f = to_float_np(gt)

        # Convert to tensors for LPIPS & GMSD
        pred_t = pred.unsqueeze(0).to(device)
        gt_t = gt.unsqueeze(0).to(device)

        # Metrics
        metrics = {}
        metrics["L1"] = float(F.l1_loss(pred_t, gt_t).cpu())
        metrics["SSIM"] = ssim(gt_np, pred_np, channel_axis=-1, data_range=255)
        metrics["LPIPS"] = float(lpips_fn(pred_t*2-1, gt_t*2-1).cpu())
        metrics["GMSD"] = float(gmsd(pred_t, gt_t).cpu())

        # Color
        metrics["ΔE2000"] = delta_e2000(pred_f, gt_f)
        metrics["Hue_Error"] = hue_error(pred_f, gt_f)

        # Sharpness / contrast
        metrics["HF_Energy_Ratio"] = high_freq_energy_ratio(pred_f, gt_f)
        metrics["Local_Contrast_Diff"] = abs(local_contrast(pred_f) - local_contrast(gt_f))

        # Artifacts
        metrics["Hist_EMD"] = histogram_emd(pred_np, gt_np)

        # No-reference
        metrics["NIQE"] = niqe_score(pred_np)

        results.append(metrics)

    df = pd.DataFrame(results)
    return df



class ImageEvaluator:
    """
    A class to evaluate image pairs with various metrics, pre-loading LPIPS.
    """
    def __init__(self, device=None):
        """
        Initializes the ImageEvaluator.

        Args:
            device: The device to use for LPIPS (e.g., "cuda" or "cpu"). 
                    Defaults to "cuda" if available, otherwise "cpu".
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Preload LPIPS once
        self.lpips_fn = lpips.LPIPS(net='alex').to(self.device)
        self.results_df = pd.DataFrame()

    def evaluate_image_pairs(self, image_pairs):
        """
        Evaluates a list of (pred, gt) image pairs and adds the results to the internal DataFrame.

        Args:
            image_pairs: A list of (pred, gt) images, where each image is a [C,H,W] tensor in [0,1].
        """
        new_results = []
        for pred, gt in image_pairs:
            pred_np = to_numpy(pred)
            gt_np = to_numpy(gt)
            pred_f = to_float_np(pred)
            gt_f = to_float_np(gt)

            # Convert to tensors for LPIPS & GMSD
            pred_t = pred.unsqueeze(0).to(self.device)
            gt_t = gt.unsqueeze(0).to(self.device)

            # Metrics
            metrics = {}
            metrics["L1"] = float(F.l1_loss(pred_t, gt_t).cpu())
            metrics["SSIM"] = ssim(gt_np, pred_np, channel_axis=-1, data_range=255)
            metrics["LPIPS"] = float(self.lpips_fn(pred_t * 2 - 1, gt_t * 2 - 1).cpu())
            metrics["GMSD"] = float(gmsd(pred_t, gt_t).cpu())

            # Color
            metrics["ΔE2000"] = delta_e2000(pred_f, gt_f)
            metrics["Hue_Error"] = hue_error(pred_f, gt_f)

            # Sharpness / contrast
            metrics["HF_Energy_Ratio"] = high_freq_energy_ratio(pred_f, gt_f)
            metrics["Local_Contrast_Diff"] = abs(local_contrast(pred_f) - local_contrast(gt_f))

            # Artifacts
            metrics["Hist_EMD"] = histogram_emd(pred_np, gt_np)

            # No-reference
            metrics["NIQE"] = niqe_score(pred_np)

            new_results.append(metrics)

        new_df = pd.DataFrame(new_results)
        self.results_df = pd.concat([self.results_df, new_df], ignore_index=True)

    def get_results(self):
        """
        Returns the internal DataFrame containing all evaluation results so far.

        Returns:
            A Pandas DataFrame with all accumulated metrics.
        """
        return self.results_df