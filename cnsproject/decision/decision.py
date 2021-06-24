"""
Module for decision making.
"""
from numpy.core import shape_base
import torch
import numpy as np

from abc import ABC, abstractmethod
from typing import List, Tuple

from ..network.neural_populations import NeuralPopulation
from ..imagetools.filters import convolve


class AbstractDecision(ABC):
    """
    Abstract class to define decision making strategy.

    Make sure to implement the abstract methods in your child class.
    """

    @abstractmethod
    def compute(self, **kwargs) -> None:
        """
        Infer the decision to be made.

        Returns
        -------
        None
            It should return the decision result.

        """
        pass

    @abstractmethod
    def update(self, **kwargs) -> None:
        """
        Update the variables after making the decision.

        Returns
        -------
        None

        """
        pass


class WinnerTakeAllDecision(AbstractDecision):
    """
    The k-Winner-Take-All decision mechanism.

    You will have to define a constructor and specify the required \
    attributes, including k, the number of winners.
    """

    def __init__(self, winners_size: int, features: torch.Tensor, populations: List[NeuralPopulation]) -> None:
        self.features = features
        self.populations = populations
        self.winners_size = winners_size
        self.winners = []
        pass

    def compute(self, input_spikes: torch.Tensor) -> None:
        """
        Infer the decision to be made.

        Returns
        -------
        None
            It should return the decision result.

        """
        if len(self.winners) == self.winners_size:
            return

        for idx, feature in enumerate(self.features):
            self.populations[idx].forward(
                torch.zeros(self.populations[idx].shape), convolve(input_spikes, feature))
            
            if idx >= len(self.winners):
                tmp = sorted(enumerate(self.populations[idx].prev_potential.flatten()), key=lambda x: x[1], reverse=True)
                for idx_prime, _ in tmp:
                    if self.populations[idx].s.flatten()[idx_prime] == 1:
                        tmp2 = np.unravel_index([idx_prime], self.populations[idx].shape)
                        position_winner = (tmp2[0][0], tmp2[1][0])
                        if not self.is_in_inhibition_position(position_winner):
                            self.winners.append(position_winner)
                            return


    def is_in_inhibition_position(self, position: Tuple[int, int]) -> bool:
        for idx, winner in enumerate(self.winners):
            y_length = len(self.features[idx])
            x_length = len(self.features[idx][0])

            if ((winner[0] - y_length < position[0] < winner[0] + y_length) and (winner[1] - x_length < position[1] < winner[1] + x_length)):
                return True
            
        return False

    def update(self, **kwargs) -> None:
        """
        Update the variables after making the decision.

        Returns
        -------
        None

        """
        pass
