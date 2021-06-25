"""
Module for encoding data into spike.
"""

from abc import ABC, abstractmethod
from typing import Optional, Callable

import torch
import numpy as np

class AbstractEncoder(ABC):
    """
    Abstract class to define encoding mechanism.

    You will define the time duration into which you want to encode the data \
    as `time` and define the time resolution as `dt`. All computations will be \
    performed on the CPU by default. To handle computation on both GPU and CPU, \
    make sure to set the device as defined in `device` attribute to all your \
    tensors. You can add any other attributes to the child classes, if needed.

    The computation procedure should be implemented in the `__call__` method. \
    Data will be passed to this method as a tensor for further computations. You \
    might need to define more parameters for this method. The `__call__`  should return \
    the tensor of spikes with the shape (time_steps, \*population.shape).

    Arguments
    ---------
    time : int
        Length of encoded tensor.
    dt : float, Optional
        Simulation time step. The default is 1.0.
    device : str, Optional
        The device to do the computations. The default is "cpu".

    """

    def __init__(
        self,
        time: int,
        dt: Optional[float] = 0.001,
        device: Optional[str] = "cpu",
        **kwargs
    ) -> None:
        self.time = time
        self.dt = dt
        self.device = device
        self.steps = int(time / dt)

    @abstractmethod
    def __call__(self, data: torch.Tensor) -> None:
        """
        Compute the encoded tensor of the given data.

        Parameters
        ----------
        data : torch.Tensor
            The data tensor to encode.

        Returns
        -------
        None
            It should return the encoded tensor.

        """
        pass


class Time2FirstSpikeEncoder(AbstractEncoder):
    """
    Time-to-First-Spike coding.

    Implement Time-to-First-Spike coding.
    """

    def __init__(
        self,
        time: int,
        dt: Optional[float] = 0.001,
        device: Optional[str] = "cpu",
        **kwargs
    ) -> None:
        super().__init__(
            time=time,
            dt=dt,
            device=device,
            **kwargs
        )

    def __call__(self, min_val: int, max_val: int, data: torch.Tensor) -> torch.Tensor:
        shape = data.shape
        coded_data = torch.zeros(self.steps, *shape)
        
        data_scaled_flatten = [
            int(min_max_scale(min_val, max_val, val) * self.steps) for val in data.flatten() ]

        for i in range(self.steps):
            spikes = torch.tensor(
                list(map(
                    lambda x: 1 if x == self.steps - i else 0,
                    data_scaled_flatten))
            ).view(shape)

            coded_data[i] = spikes

        return coded_data

class PositionEncoder(AbstractEncoder):
    """
    Position coding.

    Implement Position coding.
    """

    def __init__(
        self,
        time: int,
        neurons_number: int,
        dt: Optional[float] = 0.001,
        device: Optional[str] = "cpu",
        **kwargs
    ) -> None:
        super().__init__(
            time=time,
            dt=dt,
            device=device,
            **kwargs
        )

        self.neurons_number = neurons_number

    def __call__(self, min_val: int, max_val: int, data: torch.Tensor) -> torch.Tensor:
        neuron_distance = (max_val - min_val) / self.neurons_number
        def make_gaussian_neuron(mu, sigma):
            f = gaussian_pdf(mu, sigma)
            return lambda x: (f(x) * -self.steps) / f(mu) + self.steps
        
        gaussian_neurons = [
            make_gaussian_neuron(i * neuron_distance, neuron_distance) for i in range(self.neurons_number) ]
        
        return torch.tensor(
            list(map(
                lambda x: [f(x) for f in gaussian_neurons],
                data.flatten()))
        ).view(*data.shape, self.neurons_number)


class PoissonEncoder(AbstractEncoder):
    """
    Poisson coding.

    Implement Poisson coding.
    """

    def __init__(
        self,
        time: int,
        max_spikes: int,
        dt: Optional[float] = 0.001,
        device: Optional[str] = "cpu",
        **kwargs
    ) -> None:
        super().__init__(
            time=time,
            dt=dt,
            device=device,
            **kwargs
        )

        self.max_spikes = max_spikes

    def __call__(self, min_val: int, max_val: int, data: torch.Tensor) -> torch.Tensor:
        shape = data.shape
        coded_data = torch.zeros(self.steps, *shape)
        data_flatten = data.flatten()

        for i in range(self.steps):
            spikes = torch.tensor(
                list(map(
                    lambda x: 1 if np.random.rand() < (self.max_spikes / self.steps) * (x / max_val) else 0,
                    data_flatten))
            ).view(shape)

            coded_data[i] = spikes
        
        return coded_data


def min_max_scale(min_val: float, max_val: float, value: float) -> float:
    return (value - min_val) / (max_val - min_val)


def gaussian_pdf(mu: float, sigma: float) -> Callable[[float], float]:
    return lambda x: 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp( - (x - mu) ** 2 / (2 * sigma ** 2) )