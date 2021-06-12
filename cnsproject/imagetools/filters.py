import enum
import torch

from typing import Tuple
from numpy import pi, sqrt, cos, sin


class FilterModes(enum.Enum):
    OnCenter = 1
    OffCenter = 2


def make_axis(size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    n = int((size - 1) / 2) 
    axis_range = torch.arange(-n, n + 1)
    xs, ys = torch.meshgrid([axis_range, axis_range])

    return (xs, ys)


def gaussian(xs: torch.Tensor, ys: torch.Tensor, std: float) -> torch.Tensor:
    return torch.exp(-(xs ** 2 + ys ** 2) / (2 * std)) / (std * sqrt(2 * pi))


def make_gaussian_filter(size: int, std: float):
    xs, ys = make_axis(size)
    
    return gaussian(xs, ys, std)


def DoG(std_1: float, std_2: float, size: int) -> torch.Tensor:
    g1 = make_gaussian_filter(size, std_1)
    g2 = make_gaussian_filter(size, std_2)

    return (g1 - g2)


def gabor(lambda_: float, theta: float, sigma: float, gamma: float, size: int) -> torch.Tensor:
    xs, ys = make_axis(size)
    X = xs * cos(theta) + ys * sin(theta)
    Y = -xs * sin(theta) + ys * cos(theta)
    
    E = torch.exp(-(X ** 2 + gamma ** 2 * Y ** 2) / (2 * sigma ** 2))
    C = torch.cos((2 * pi / lambda_) * X)

    return (E * C)


def convolve(image: torch.Tensor, filter: torch.Tensor, mode: FilterModes = FilterModes.OnCenter):
    filter_prime = filter * -1 if mode == FilterModes.OffCenter else filter

    image_rows = image.shape[0]
    image_columns = image.shape[1]
    
    filter_rows = filter.shape[0]
    filter_columns = filter.shape[1]
    
    result = torch.zeros(image_rows - filter_rows + 1, image_columns - filter_columns + 1)
    rows = result.shape[0]
    columns = result.shape[1]
    for i in range(rows):
        for j in range(columns):
            result[i, j] = (filter_prime * image[i: i + filter_rows, j: j + filter_columns]).sum()

    return result