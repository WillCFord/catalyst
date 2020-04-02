from typing import Sequence

import numpy as np

import torch

from catalyst import utils
from catalyst.contrib.utils.image import _IMAGENET_MEAN, _IMAGENET_STD

# <------- taken from torchvision - https://github.com/pytorch/vision ------->


def _is_numpy(img: np.ndarray) -> bool:
    return isinstance(img, np.ndarray)


def _is_numpy_image(img: np.ndarray) -> bool:
    return img.ndim in {2, 3}


def normalize(
    tensor: torch.Tensor,
    mean: Sequence[float] = _IMAGENET_MEAN,
    std: Sequence[float] = _IMAGENET_STD,
    inplace: bool = False,
) -> torch.Tensor:
    """Normalize a tensor image with mean and standard deviation.

    .. note::
        This transform acts out of place by default, i.e.,
        it does not mutates the input tensor.

    Args:
        tensor (torch.Tensor): tensor image of size (C, H, W) to be normalized
        mean (Sequence[float]): Sequence of means for each channel
        std (Sequence[float]): Sequence of standard deviations for each channel
        inplace(bool,optional): bool to make this operation inplace

    Returns:
        torch.Tensor: normalized Tensor image
    """
    if not torch.is_tensor(tensor):
        raise TypeError(
            f"tensor should be a torch tensor. Got {type(tensor)}."
        )

    if tensor.ndimension() != 3:
        raise ValueError(
            f"Expected tensor to be a tensor image of size (C, H, W)."
            f"Got tensor.size() = {tensor.size()}"
        )

    if not inplace:
        tensor = tensor.clone()

    dtype = tensor.dtype
    mean = torch.as_tensor(mean, dtype=dtype, device=tensor.device)
    std = torch.as_tensor(std, dtype=dtype, device=tensor.device)
    if (std == 0).any():
        raise ValueError(
            f"std evaluated to zero after conversion to {dtype},"
            f" leading to division by zero."
        )
    if mean.ndim == 1:
        mean = mean[:, None, None]
    if std.ndim == 1:
        std = std[:, None, None]
    tensor.sub_(mean).div_(std)
    return tensor


def to_tensor(pic: np.ndarray) -> torch.Tensor:
    """Convert a ``numpy.ndarray`` to tensor

    Args:
        pic (numpy.ndarray): image to be converted to tensor

    Returns:
        torch.Tensor: converted image
    """
    if not _is_numpy(pic):
        raise TypeError(f"pic should be PIL Image or ndarray. Got {type(pic)}")
    elif not _is_numpy_image(pic):
        raise ValueError(
            f"pic should be 2/3 dimensional. Got {pic.ndim} dimensions."
        )

    if pic.ndim == 2:
        pic = pic[:, :, None]

    img = torch.from_numpy(pic.transpose((2, 0, 1)))
    # backward compatibility
    if isinstance(img, torch.ByteTensor):
        return img.float().div(255)
    return img


# <------- taken from torchvision - https://github.com/pytorch/vision ------->


def test_imread():
    """Tests ``imread`` functionality."""
    jpg_rgb_uri = (
        "https://raw.githubusercontent.com/catalyst-team/catalyst-pics/master"
        "/test_images/catalyst_icon.jpg"
    )
    jpg_grs_uri = (
        "https://raw.githubusercontent.com/catalyst-team/catalyst-pics/master"
        "/test_images/catalyst_icon_grayscale.jpg"
    )
    png_rgb_uri = (
        "https://raw.githubusercontent.com/catalyst-team/catalyst-pics/master"
        "/test_images/catalyst_icon.png"
    )
    png_grs_uri = (
        "https://raw.githubusercontent.com/catalyst-team/catalyst-pics/master"
        "/test_images/catalyst_icon_grayscale.png"
    )

    for uri in [jpg_rgb_uri, jpg_grs_uri, png_rgb_uri, png_grs_uri]:
        img = utils.imread(uri)
        assert img.shape == (400, 400, 3)
        img = utils.imread(uri, grayscale=True)
        assert img.shape == (400, 400, 1)


def test_tensor_to_ndimage():
    """Tests ``tensor_to_ndimage`` functionality."""
    orig_images = np.random.randint(0, 255, (2, 20, 10, 3), np.uint8)

    torch_images = torch.stack(
        [
            normalize(to_tensor(im), _IMAGENET_MEAN, _IMAGENET_STD)
            for im in orig_images
        ],
        dim=0,
    )

    byte_images = utils.tensor_to_ndimage(torch_images, dtype=np.uint8)
    float_images = utils.tensor_to_ndimage(torch_images, dtype=np.float32)

    assert np.allclose(byte_images, orig_images)
    assert np.allclose(float_images, orig_images / 255, atol=1e-3, rtol=1e-3)

    assert np.allclose(
        utils.tensor_to_ndimage(torch_images[0]),
        orig_images[0] / 255,
        atol=1e-3,
        rtol=1e-3,
    )
