"""
Module for visualization and plotting.
"""
import matplotlib.pyplot as plt
from typing import List, Callable

from ..network.neural_populations import LIFPopulation


def plot_lif(lif: LIFPopulation, potentials: List[float]):
    steps = len(potentials)
    potential_times = [i * lif.dt for i in range(steps)]
    plt.plot(potential_times, potentials, c='g')
    plt.scatter(
        lif.spike_times,
        [lif.threshold for _ in range(len(lif.spike_times))],
        c='r',
        label='spikes')

    plt.xlabel('Time')
    plt.ylabel('Potential')
    plt.legend()
    plt.show()


def plot_current(current: Callable[[float], float], steps: int, dt: float):
    times = [dt * i for i in range(steps)]
    current_values = [current(dt * i) for i in range(steps)]
    plt.plot(times, current_values)
    plt.xlabel('Time')
    plt.ylabel('Current')
    plt.show()


def plot_FI_for_lif(lif: LIFPopulation, simulation_seconds=5, current_range=50):
    freqs = []
    steps = int(simulation_seconds / lif.dt)
    for i in range(current_range):
        lif.refractory_and_reset()
        lif.spike_times = []
        lif.current = lambda t: i
        for _ in range(steps):
            lif.forward(None)
        
        frequency = len(lif.spike_times) / simulation_seconds
        freqs.append(frequency)

    plt.plot([i for i in range(current_range)], freqs)
    plt.xlabel('I(t)')
    plt.ylabel('Frequency')
    plt.show()
