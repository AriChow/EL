import torch
import torch.nn as nn
import torch.nn.functional as F
from EL.utils.utils import gumbel_softmax_sample


class GumbelSoftmaxLayer(nn.Module):
    def __init__(self,
                 temperature: float = 1.0,
                 trainable_temperature: bool = False,
                 straight_through: bool = False):
        super(GumbelSoftmaxLayer, self).__init__()
        self.straight_through = straight_through

        if not trainable_temperature:
            self.temperature = temperature
        else:
            self.temperature = torch.nn.Parameter(
                torch.tensor([temperature]), requires_grad=True)

    def forward(self, logits: torch.Tensor):
        return gumbel_softmax_sample(logits, self.temperature, self.training, self.straight_through)


class GumbelSoftmaxWrapper(nn.Module):
    """
    Gumbel-Softmax Wrapper for an agent that outputs a single symbol. Assumes that during the forward pass,
    the agent returns log-probabilities over the potential output symbols. During training, the wrapper
    transforms them into a sample from the Gumbel Softmax (GS) distribution; eval-time it returns greedy one-hot encoding
    of the same shape.

    >>> inp = torch.zeros((4, 10)).uniform_()
    >>> outp = GumbelSoftmaxWrapper(nn.Linear(10, 2))(inp)
    >>> torch.allclose(outp.sum(dim=-1), torch.ones_like(outp.sum(dim=-1)))
    True
    >>> outp = GumbelSoftmaxWrapper(nn.Linear(10, 2), straight_through=True)(inp)
    >>> torch.allclose(outp.sum(dim=-1), torch.ones_like(outp.sum(dim=-1)))
    True
    >>> (max_value, _), (min_value, _) = outp.max(dim=-1), outp.min(dim=-1)
    >>> (max_value == 1.0).all().item() == 1 and (min_value == 0.0).all().item() == 1
    True
    """

    def __init__(self,
                 agent,
                 temperature=1.0,
                 trainable_temperature=False,
                 straight_through=False):
        """
        :param agent: The agent to be wrapped. agent.forward() has to output log-probabilities over the vocabulary
        :param temperature: The temperature of the Gumbel Softmax distribution
        :param trainable_temperature: If set to True, the temperature becomes a trainable parameter of the model
        :params straight_through: Whether straigh-through Gumbel Softmax is used
        """
        super(GumbelSoftmaxWrapper, self).__init__()
        self.agent = agent
        self.temperature = temperature
        self.straight_through = straight_through
        self.gs_layer = GumbelSoftmaxLayer(temperature=self.temperature, trainable_temperature=trainable_temperature,
                                           straight_through=straight_through)

    def forward(self, *args, **kwargs):
        logits = self.agent(*args, **kwargs)
        sample = self.gs_layer(logits)
        return sample


class SymbolGameGS(nn.Module):
    """
    Implements one-symbol Sender/Receiver game. The loss must be differentiable wrt the parameters of the agents.
    Typically, this assumes Gumbel Softmax relaxation of the communication channel.
    >>> class Receiver(nn.Module):
    ...     def forward(self, x, _input=None):
    ...         return x

    >>> receiver = Receiver()
    >>> sender = nn.Sequential(nn.Linear(10, 10), nn.LogSoftmax(dim=1))

    >>> def mse_loss(sender_input, _1, _2, receiver_output, _3):
    ...     return (sender_input - receiver_output).pow(2.0).mean(dim=1), {}

    >>> game = SymbolGameGS(sender=sender, receiver=Receiver(), loss=mse_loss)
    >>> forward_result = game(torch.ones((2, 10)), None) #  the second argument is labels, we don't need any
    >>> forward_result[1]
    {}
    >>> (forward_result[0] > 0).item()
    1
    """
    def __init__(self, sender, receiver, loss):
        """
        :param sender: Sender agent. sender.forward() has to output log-probabilities over the vocabulary.
        :param receiver: Receiver agent. receiver.forward() has to accept two parameters: message and receiver_input.
        `message` is shaped as (batch_size, vocab_size).
        :param loss: Callable that outputs differentiable loss, takes the following parameters:
          * sender_input: input to Sender (comes from dataset)
          * message: message sent from Sender
          * receiver_input: input to Receiver from dataset
          * receiver_output: output of Receiver
          * labels: labels that come from dataset
        """
        super(SymbolGameGS, self).__init__()
        self.sender = sender
        self.receiver = receiver
        self.loss = loss

    def forward(self, sender_input, labels=None, receiver_input=None):
        message = self.sender(sender_input)
        receiver_output = self.receiver(message, receiver_input)
        if labels is None:
            return receiver_output
        loss, rest_info = self.loss(sender_input, message, receiver_input, receiver_output, labels)
        for k, v in rest_info.items():
            if hasattr(v, 'mean'):
                rest_info[k] = v.mean().item()

        return loss.mean(), rest_info


class SymbolReceiverWrapper(nn.Module):
    """
    An optional wrapper for single-symbol Receiver, both Gumbel-Softmax and Reinforce. Receives a message, embeds it,
    and passes to the wrapped agent.
    """
    def __init__(self, agent, vocab_size, agent_input_size):
        super(SymbolReceiverWrapper, self).__init__()
        self.agent = agent
        self.embedding = RelaxedEmbedding(vocab_size, agent_input_size)

    def forward(self, message, input=None):
        embedded_message = self.embedding(message)
        return self.agent(embedded_message, input)
