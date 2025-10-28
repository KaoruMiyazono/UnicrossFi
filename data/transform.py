import random
import numpy as np
import torch
from PIL import Image, ImageFilter


def random_applied_trans(csi, transform_names, p=0.5, namespace=None):
    """
    Randomly applies a list of CSI tensor-based transform functions with probability p.

    Args:
        csi (torch.Tensor): Input CSI data.
        transform_names (list[str]): List of transform function names as strings.
        p (float): Probability of applying the composed transform.
        namespace (dict): Optional namespace where transform functions are defined (e.g., globals()).

    Returns:
        torch.Tensor: Possibly transformed CSI data.
    """
    if namespace is None:
        namespace = globals()

    # Resolve string function names to actual functions
    transforms = []
    for name in transform_names:
        if name not in namespace or not callable(namespace[name]):
            raise ValueError(f"Transform '{name}' not found or is not callable in the provided namespace.")
        transforms.append(namespace[name])

    if random.random() < p:
        for t in transforms:
            csi = t(csi)

    return csi



def gaussian_blur(img, sigma_min=0.1, sigma_max=2.0, p=0.5):
    """
    Apply Gaussian Blur to the input image with a given probability.

    Args:
        img (PIL.Image): Input image.
        sigma_min (float): Minimum standard deviation for Gaussian kernel.
        sigma_max (float): Maximum standard deviation for Gaussian kernel.
        p (float): Probability of applying the transform.

    Returns:
        PIL.Image: Transformed image.
    """
    assert 0 <= p <= 1.0, f'The probability p must be in [0,1], but got {p}'
    
    if np.random.rand() > p:
        return img
    
    sigma = np.random.uniform(sigma_min, sigma_max)
    return img.filter(ImageFilter.GaussianBlur(radius=sigma))


def no_action(csi):
    """
    No augmentation. This is used for testing or when no preprocessing is desired.

    Args:
        csi (Tensor or ndarray): Input CSI data.

    Returns:
        Same as input, unchanged.
    """
    return csi




def jitter_csi(csi, sigma=0.002):
    """
    Add Gaussian noise to the original CSI data.

    Args:
        csi (torch.Tensor): Input CSI data of shape [A, C, T, 2].
        sigma (float): Standard deviation of the Gaussian noise.

    Returns:
        torch.Tensor: CSI data with added Gaussian noise.
    """
    noise = torch.from_numpy(np.random.normal(loc=0, scale=sigma, size=csi.shape)).type(csi.dtype)
    return csi + noise


def permutation_csi(csi, num_segments):
    """
    Slice the input CSI data along the time axis into multiple segments, 
    and then randomly permute them.

    Args:
        csi (torch.Tensor): Input CSI data of shape [A, C, T, 2].
        num_segments (int): Number of segments to divide the time axis into.

    Returns:
        torch.Tensor: Permuted CSI data.
    """
    time_length = csi.shape[2]
    
    # Generate segment boundaries
    segment_points = np.sort(np.random.choice(time_length, size=num_segments, replace=False))
    
    # Split and permute along the time dimension
    csi_np = csi.numpy()
    splitted = np.split(csi_np, segment_points, axis=2)
    random.shuffle(splitted)
    concat = np.concatenate(splitted, axis=2)

    # Convert back to tensor
    return torch.from_numpy(concat).type(csi.dtype)



def scaling_csi(csi, sigma):
    """
    Scale the CSI signal with a randomly generated scaling factor.

    Args:
        csi (torch.Tensor): Input CSI data, shape can be [A, C, T, 2].
        sigma (float): Standard deviation of the Gaussian distribution used for scaling.

    Returns:
        torch.Tensor: Scaled CSI data.
    """    
    # Generate scaling factor
    scaling_factor = torch.from_numpy(
        np.random.normal(loc=1.0, scale=sigma, size=(csi.shape[0],csi.shape[1],1,1))
    ).type(csi.dtype).to(csi.device)

    # Apply scaling
    csi_scaled = csi * scaling_factor

    return csi_scaled
    

def inversion_csi(csi):
    """
    This transformation multiplies the input CSI data by -1.

    Args:
        csi (Tensor): Input CSI tensor of shape [...].

    Returns:
        Tensor: CSI tensor after inversion.
    """
    return csi * -1


def time_flipping_csi(csi):
    """
    This transformation reverses the input CSI data along the time axis.

    Args:
        csi (Tensor): Input CSI tensor of shape [A,C, T,2] or [..., T].

    Returns:
        Tensor: CSI tensor after time flipping.
    """
    return torch.flip(csi, dims=[2])
