"""
Module for connections between neural populations.
"""

from abc import ABC, abstractmethod
from typing import Union, Sequence, Tuple

import random
from numpy.core.fromnumeric import size
import torch

from .neural_populations import NeuralPopulation
from ..imagetools.filters import convolve


MAX_RANDOM_WEIGHT = 0.4
MIN_WEIGHT = 0.1

class AbstractConnection(ABC, torch.nn.Module):
    """
    Abstract class for implementing connections.

    Make sure to implement the `compute`, `update`, and `reset_state_variables`\
    methods in your child class.

    You will need to define the populations you want to connect as `pre` and `post`.\
    In case of learning, you will need to define the learning rate (`lr`) and the \
    learning rule to follow. Attribute `w` is reserved for synaptic weights.\
    However, it has not been predefined or allocated, as it depends on the \
    pattern of connectivity. So make sure to define it in child class initializations \
    appropriately to indicate the pattern of connectivity. The default range of \
    each synaptic weight is [0, 1] but it can be controlled by `wmin` and `wmax`. \
    Synaptic strengths might decay in time and do not last forever. To define \
    the decay rate of the synaptic weights, use `weight_decay` attribute. Also, \
    if you want to control the overall input synaptic strength to each neuron, \
    use `norm` argument to normalize the synaptic weights.

    In case of learning, you have to implement the methods `compute` and `update`. \
    You will use the `compute` method to calculate the activity of post-synaptic \
    population based on the pre-synaptic one. Update of weights based on the \
    learning rule will be implemented in the `update` method. If you find this \
    architecture mind-bugling, try your own architecture and make sure to redefine \
    the learning rule architecture to be compatible with this new architecture \
    of yours.

    Arguments
    ---------
    pre : NeuralPopulation
        The pre-synaptic neural population.
    post : NeuralPopulation
        The post-synaptic neural population.
    lr : float or (float, float), Optional
        The learning rate for training procedure. If a tuple is given, the first
        value defines potentiation learning rate and the second one depicts\
        the depression learning rate. The default is None.
    weight_decay : float, Optional
        Define rate of decay in synaptic strength. The default is 0.0.

    Keyword Arguments
    -----------------
    learning_rule : LearningRule
        Define the learning rule by which the network will be trained. The\
        default is NoOp (see learning/learning_rules.py for more details).
    wmin : float
        The minimum possible synaptic strength. The default is 0.0.
    wmax : float
        The maximum possible synaptic strength. The default is 1.0.
    norm : float
        Define a normalization on input signals to a population. If `None`,\
        there is no normalization. The default is None.

    """

    def __init__(
        self,
        pre: NeuralPopulation,
        post: NeuralPopulation,
        lr: Union[float, Sequence[float]] = None,
        weight_decay: float = 0.0,
        **kwargs
    ) -> None:
        super().__init__()

        assert isinstance(pre, NeuralPopulation), \
            "Pre is not a NeuralPopulation instance"
        assert isinstance(post, NeuralPopulation), \
            "Post is not a NeuralPopulation instance"

        self.pre = pre
        self.post = post

        self.weight_decay = weight_decay

        from ..learning.learning_rules import NoOp

        learning_rule = kwargs.get('learning_rule', NoOp)

        self.learning_rule = learning_rule(
            connection=self,
            lr=lr,
            weight_decay=weight_decay,
            **kwargs
        )
        self.wmin = kwargs.get('wmin', 0.)
        self.wmax = kwargs.get('wmax', 1.)
        self.norm = kwargs.get('norm', None)

    @abstractmethod
    def compute(self, s: torch.Tensor) -> None:
        """
        Compute the post-synaptic neural population activity based on the given\
        spikes of the pre-synaptic population.

        Parameters
        ----------
        s : torch.Tensor
            The pre-synaptic spikes tensor.

        Returns
        -------
        None

        """
        pass

    @abstractmethod
    def update(self, **kwargs) -> None:
        """
        Compute connection's learning rule and weight update.

        Keyword Arguments
        -----------------
        learning : bool
            Whether learning is enabled or not. The default is True.
        mask : torch.ByteTensor
            Define a mask to determine which weights to clamp to zero.

        Returns
        -------
        None

        """
        learning = kwargs.get("learning", True)

        if learning:
            self.learning_rule.update(**kwargs)

        mask = kwargs.get("mask", None)
        if mask is not None:
            self.weight.masked_fill_(mask, 0)

    @abstractmethod
    def reset_state_variables(self) -> None:
        """
        Reset all internal state variables.

        Returns
        -------
        None

        """
        pass


class DenseConnection(AbstractConnection):
    """
    Specify a fully-connected synapse between neural populations.

    Implement the dense connection pattern following the abstract connection\
    template.
    """

    def __init__(
        self,
        pre: NeuralPopulation,
        post: NeuralPopulation,
        lr: Union[float, Sequence[float]] = None,
        weight_decay: float = 0.0,
        **kwargs
    ) -> None:
        super().__init__(
            pre=pre,
            post=post,
            lr=lr,
            weight_decay=weight_decay,
            **kwargs
        )
        
        self.register_buffer('weight', torch.rand(*post.shape, *pre.shape) * MAX_RANDOM_WEIGHT + MIN_WEIGHT)
        if pre.is_inhibitory:
            self.weight = -self.weight

    def compute(self, s: torch.Tensor) -> torch.Tensor:
        return self.weight @ (1.0 * s)

    def update(self, **kwargs) -> None:
        super().update(**kwargs)

    def reset_state_variables(self) -> None:
        """
        TODO.

        Reset all the state variables of the connection.
        """
        pass

    def __str__(self):
        return 'Dense Connection'


class RandomConnection(AbstractConnection):
    """
    Specify a random synaptic connection between neural populations.

    Implement the random connection pattern following the abstract connection\
    template.
    """

    def __init__(
        self,
        pre: NeuralPopulation,
        post: NeuralPopulation,
        connection_size: int,
        lr: Union[float, Sequence[float]] = None,
        weight_decay: float = 0.0,
        **kwargs
    ) -> None:
        super().__init__(
            pre=pre,
            post=post,
            lr=lr,
            weight_decay=weight_decay,
            **kwargs
        )

        self.register_buffer('connection_size', torch.tensor(connection_size))
        self.register_buffer('mask', self.compute_mask())
        self.register_buffer('weight', torch.rand(*post.shape, *pre.shape) * self.mask * MAX_RANDOM_WEIGHT + MIN_WEIGHT)
        if pre.is_inhibitory:
            self.weight = -self.weight

    def compute_mask(self) -> torch.Tensor:
        mask = torch.zeros(*self.post.shape, *self.pre.shape)
        for idx, row in enumerate(mask):
            connections_idx = random.sample(range(len(row)), self.connection_size)
            mask[idx, connections_idx] = torch.ones(self.connection_size)
        
        return mask

    def compute(self, s: torch.Tensor) -> None:
        return self.weight @ (1.0 * s)

    def update(self, **kwargs) -> None:
        super().update(**kwargs)

    def reset_state_variables(self) -> None:
        """
        TODO.

        Reset all the state variables of the connection.
        """
        pass

    def __str__(self):
        return 'Random connection with {} presynaptic connections'.format(self.connection_size)


class ConvolutionalConnection(AbstractConnection):
    """
    Specify a convolutional synaptic connection between neural populations.

    Implement the convolutional connection pattern following the abstract\
    connection template.
    """

    def __init__(
        self,
        pre: NeuralPopulation,
        post: NeuralPopulation,
        kernel: torch.Tensor,
        lr: Union[float, Sequence[float]] = None,
        weight_decay: float = 0.0,
        **kwargs
    ) -> None:
        super().__init__(
            pre=pre,
            post=post,
            lr=lr,
            weight_decay=weight_decay,
            **kwargs
        )

        self.register_buffer('kernel', kernel)

    def compute(self, s: torch.Tensor) -> torch.Tensor:
        if s.shape != self.pre.shape:
            raise Exception('Spikes shape is diffrent from pre shape!')
        
        return convolve(s, torch.flipud(torch.fliplr(self.kernel)))

    def update(self, **kwargs) -> None:
        super().update(**kwargs)

    def reset_state_variables(self) -> None:
        """
        TODO.

        Reset all the state variables of the connection.
        """
        pass


class PoolingConnection(AbstractConnection):
    """
    Specify a pooling synaptic connection between neural populations.

    Implement the pooling connection pattern following the abstract connection\
    template. Consider a parameter for defining the type of pooling.

    Note: The pooling operation does not support learning. You might need to\
    make some modifications in the defined structure of this class.
    """

    def __init__(
        self,
        pre: NeuralPopulation,
        post: NeuralPopulation,
        size: int,
        stride: int,
        lr: Union[float, Sequence[float]] = None,
        weight_decay: float = 0.0,
        **kwargs
    ) -> None:
        super().__init__(
            pre=pre,
            post=post,
            lr=lr,
            weight_decay=weight_decay,
            **kwargs
        )

        self.size = size
        self.stride = stride  
        self.set_output_shape_and_mask()

    def set_output_shape_and_mask(self):
        self.output_shape = calculate_pooling_post_population_size(self.size, self.stride, self.pre.shape)
        if self.post.shape != self.output_shape:
            raise Exception('Wrong post population shape!')

        self.output_mask = torch.ones(self.output_shape[0], self.output_shape[1])

    def compute(self, s: torch.Tensor) -> torch.Tensor:
        result = torch.zeros(self.output_shape)
        for i in range(self.output_shape[0]):
            for j in range(self.output_shape[1]):
                x = i * self.stride
                y = j * self.stride
                if s[x: x + self.size, y: y + self.size].sum() >= 1:
                    if self.output_mask[i][j] != 0:
                        result[i][j] = 1
                        self.output_mask[i][j] = 0
        
        return result

    def update(self, **kwargs) -> None:
        super().update(**kwargs)

    def reset_state_variables(self) -> None:
        """
        TODO.

        Reset all the state variables of the connection.
        """
        pass


def calculate_pooling_post_population_size(size: int, stride: int, pre_shape: torch.Size) -> Tuple[int, int]:
    output_rows = int((pre_shape[0] - size) / stride) + 1
    out_put_columns = int((pre_shape[1] - size) / stride) + 1
    
    return (output_rows, out_put_columns)
