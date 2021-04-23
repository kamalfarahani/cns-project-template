"""
Module for visualization and plotting.
"""
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Callable
from functools import reduce
from operator import add

from ..network.neural_populations import NeuralPopulation


def get_random_rgb() -> np.ndarray:
    r = random.random()
    b = random.random()
    g = random.random()
    color = (r, g, b)
    return np.array(color).reshape(1,-1)


def get_spiked_neurons(spikes: torch.Tensor) -> torch.Tensor:
    spiked_neurons = list(map(
                lambda x: x[0],
                filter(
                    lambda x: x[1] != 0,
                    enumerate(spikes))))
    
    return torch.tensor(spiked_neurons)


def plot_activity(populatins_spikes: List[torch.Tensor], dt: float):
    steps = len(populatins_spikes[0])
    population_size = reduce(lambda acc, pop: acc + len(pop[0]), populatins_spikes, 0)
    activities = []
    for step in range(steps):
        active_neurons = reduce(
            add,
            map(
                lambda spikes: len(get_spiked_neurons(torch.flatten(spikes[step]))),
                populatins_spikes))
            
        activities.append(active_neurons / population_size)
    
    plt.plot([dt * i for i in range(steps)], activities)
    plt.xlabel('Time')
    plt.ylabel('Activity')
    plt.show()


def raster_plot(populatins_spikes: List[torch.Tensor], dt: float):
    acc = 0
    for spikes_per_step in populatins_spikes:
        color = get_random_rgb()
        for step, spikes in enumerate(spikes_per_step):
            spikes_flatten = torch.flatten(spikes)
            spiked_neurons = get_spiked_neurons(spikes_flatten)
            plot_neuron_index = list(map(lambda x: x + acc, spiked_neurons))
            plt.scatter(
                [dt * step] * len(spiked_neurons),
                plot_neuron_index,
                c=color, s=[1] * len(spiked_neurons))
    
        acc = acc + len(spikes_flatten)
    
    plt.xlabel('Time')
    plt.ylabel('Raster Activity')
    plt.show()


def plot_current(currents: torch.Tensor, steps: int, dt: float) -> None:
    times = [dt * i for i in range(steps)]
    plt.plot(times, currents)
    plt.xlabel('Time')
    plt.ylabel('Current')
    plt.show()


def plot_adaptation(adaptations: List[float], dt):
    times = [dt * i for i in range(len(adaptations))]
    plt.plot(times, adaptations, c='r')
    plt.xlabel('time')
    plt.ylabel('adaptation value')
    plt.show()