"""
Module for learning rules.
"""

from abc import ABC
from typing import Union, Optional, Sequence

import numpy as np
import torch

from ..network.connections import AbstractConnection


class LearningRule(ABC):
    """
    Abstract class for defining learning rules.

    Each learning rule will be applied on a synaptic connection defined as \
    `connection` attribute. It possesses learning rate `lr` and weight \
    decay rate `weight_decay`. You might need to define more parameters/\
    attributes to the child classes.

    Implement the dynamics in `update` method of the classes. Computations \
    for weight decay and clamping the weights has been implemented in the \
    parent class `update` method. So do not invent the wheel again and call \
    it at the end  of the child method.

    Arguments
    ---------
    connection : AbstractConnection
        The connection on which the learning rule is applied.
    lr : float or sequence of float, Optional
        The learning rate for training procedure. If a tuple is given, the first
        value defines potentiation learning rate and the second one depicts\
        the depression learning rate. The default is None.
    weight_decay : float
        Define rate of decay in synaptic strength. The default is 0.0.

    """

    def __init__(
        self,
        connection: AbstractConnection,
        lr: Optional[Union[float, Sequence[float]]] = None,
        weight_decay: float = 0.,
        **kwargs
    ) -> None:
        if lr is None:
            lr = [0., 0.]
        elif isinstance(lr, float) or isinstance(lr, int):
            lr = [lr, lr]

        self.lr = torch.tensor(lr, dtype=torch.float)

        self.weight_decay = 1 - weight_decay if weight_decay else 1.

        self.connection = connection

    def update(self) -> None:
        """
        Abstract method for a learning rule update.

        Returns
        -------
        None

        """
        if self.weight_decay:
            self.connection.weight *= self.weight_decay

        if (
            self.connection.wmin != -np.inf or self.connection.wmax != np.inf
        ) and not isinstance(self.connection, NoOp):
            self.connection.weight.clamp_(self.connection.wmin,
                                     self.connection.wmax)


class NoOp(LearningRule):
    """
    Learning rule with no effect.

    Arguments
    ---------
    connection : AbstractConnection
        The connection on which the learning rule is applied.
    lr : float or sequence of float, Optional
        The learning rate for training procedure. If a tuple is given, the first
        value defines potentiation learning rate and the second one depicts\
        the depression learning rate. The default is None.
    weight_decay : float
        Define rate of decay in synaptic strength. The default is 0.0.

    """

    def __init__(
        self,
        connection: AbstractConnection,
        lr: Optional[Union[float, Sequence[float]]] = None,
        weight_decay: float = 0.,
        **kwargs
    ) -> None:
        super().__init__(
            connection=connection,
            lr=lr,
            weight_decay=weight_decay,
            **kwargs
        )

    def update(self, **kwargs) -> None:
        """
        Only take care about synaptic decay and possible range of synaptic
        weights.

        Returns
        -------
        None

        """
        super().update()


class STDP(LearningRule):
    """
    Spike-Time Dependent Plasticity learning rule.

    Implement the dynamics of STDP learning rule.You might need to implement\
    different update rules based on type of connection.
    """

    def __init__(
        self,
        connection: AbstractConnection,
        lr: Optional[Union[float, Sequence[float]]] = None,
        weight_decay: float = 0.,
        **kwargs
    ) -> None:
        super().__init__(
            connection=connection,
            lr=lr,
            weight_decay=weight_decay,
            **kwargs
        )

    def update(self, **kwargs) -> None:
        pre = self.connection.pre
        post = self.connection.post
        dt = pre.dt
        dw = ( -self.lr[0] * post.traces.view(*post.shape, 1) @ pre.s.view(1, *pre.shape).float() + 
                (self.lr[1] * pre.traces.view(*pre.shape, 1) @ post.s.view(1, *post.shape).float()).transpose(0, 1) ) * dt

        self.connection.weight += dw

        super().update()


class FlatSTDP(LearningRule):
    """
    Flattened Spike-Time Dependent Plasticity learning rule.

    Implement the dynamics of Flat-STDP learning rule.You might need to implement\
    different update rules based on type of connection.
    """

    def __init__(
        self,
        connection: AbstractConnection,
        lr: Optional[Union[float, Sequence[float]]] = None,
        weight_decay: float = 0.,
        trace_window_steps: int = 100,
        **kwargs
    ) -> None:
        super().__init__(
            connection=connection,
            lr=lr,
            weight_decay=weight_decay,
            **kwargs
        )

        self.trace_window_steps = trace_window_steps
        self.pre_traces = torch.zeros(trace_window_steps, *connection.pre.shape)
        self.pre_traces_index = 0
        self.post_traces = torch.zeros(trace_window_steps, *connection.post.shape)
        self.post_traces_index = 0
    
    def add_pre_trace(self, trace):
        self.pre_traces_index = self.pre_traces_index % self.trace_window_steps
        self.pre_traces[self.pre_traces_index] = trace
        self.pre_traces_index += 1
    
    def add_post_trace(self, trace):
        self.post_traces_index = self.post_traces_index % self.trace_window_steps
        self.post_traces[self.post_traces_index] = trace
        self.post_traces_index += 1

    def update(self, **kwargs) -> None:
        pre = self.connection.pre
        self.add_pre_trace(pre.s)
        pre_traces = sum(self.pre_traces)

        post = self.connection.post
        self.add_post_trace(post.s)
        post_traces = sum(self.post_traces)
        
        dt = pre.dt
        dw = ( -self.lr[0] * post_traces.view(*post.shape, 1) @ pre.s.view(1, *pre.shape).float() + 
                (self.lr[1] * pre_traces.view(*pre.shape, 1) @ post.s.view(1, *post.shape).float()).transpose(0, 1) ) * dt

        self.connection.weight += dw

        super().update()


class RSTDP(LearningRule):
    """
    Reward-modulated Spike-Time Dependent Plasticity learning rule.

    Implement the dynamics of RSTDP learning rule. You might need to implement\
    different update rules based on type of connection.
    """

    def __init__(
        self,
        connection: AbstractConnection,
        lr: Optional[Union[float, Sequence[float]]] = None,
        weight_decay: float = 0.,
        **kwargs
    ) -> None:
        super().__init__(
            connection=connection,
            lr=lr,
            weight_decay=weight_decay,
            **kwargs
        )
        """
        TODO.

        Consider the additional required parameters and fill the body\
        accordingly.
        """

    def update(self, **kwargs) -> None:
        """
        TODO.

        Implement the dynamics and updating rule. You might need to call the
        parent method. Make sure to consider the reward value as a given keyword
        argument.
        """
        pass


class FlatRSTDP(LearningRule):
    """
    Flattened Reward-modulated Spike-Time Dependent Plasticity learning rule.

    Implement the dynamics of Flat-RSTDP learning rule. You might need to implement\
    different update rules based on type of connection.
    """

    def __init__(
        self,
        connection: AbstractConnection,
        lr: Optional[Union[float, Sequence[float]]] = None,
        weight_decay: float = 0.,
        **kwargs
    ) -> None:
        super().__init__(
            connection=connection,
            lr=lr,
            weight_decay=weight_decay,
            **kwargs
        )
        """
        TODO.

        Consider the additional required parameters and fill the body\
        accordingly.
        """

    def update(self, **kwargs) -> None:
        """
        TODO.

        Implement the dynamics and updating rule. You might need to call the
        parent method. Make sure to consider the reward value as a given keyword
        argument.
        """
        pass
