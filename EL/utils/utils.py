import torch
from typing import Any, Optional
from torch.distributions import RelaxedOneHotCategorical
from collections import defaultdict


def load_model_weights(model, old_model_path):
    pretrained_weights = torch.load(old_model_path)
    pretrained_weights_items = list(pretrained_weights.items())

    model_state_dict = model.state_dict()
    count = 0
    for key, value in model_state_dict.items():
        layer_name, weights = pretrained_weights_items[count]
        model_state_dict[key] = weights
        count += 1

    model.load_state_dict(model_state_dict)
    return model

def gumbel_softmax_sample(logits, temperature=1.0, training=True, straight_through=False):
    size = logits.size()
    if not training:
        indexes = logits.argmax(dim=-1)
        one_hot = torch.zeros_like(logits).view(-1, size[-1])
        one_hot.scatter_(1, indexes.view(-1, 1), 1)
        one_hot = one_hot.view(*size)
        return one_hot

    sample = RelaxedOneHotCategorical(
        logits=logits, temperature=temperature).rsample()

    if straight_through:
        size = sample.size()
        indexes = sample.argmax(dim=-1)
        hard_sample = torch.zeros_like(sample).view(-1, size[-1])
        hard_sample.scatter_(1, indexes.view(-1, 1), 1)
        hard_sample = hard_sample.view(*size)

        sample = sample + (hard_sample - sample).detach()
    return sample


def move_to(x: Any, device: torch.device) \
        -> Any:
    """
    Simple utility function that moves a tensor or a dict/list/tuple of (dict/list/tuples of ...) tensors to a specified device, recursively.
    :param x: tensor, list, tuple, or dict with values that are lists, tuples or dicts with values of ...
    :param device: device to be moved to
    :return: Same as input, but with all tensors placed on device. Non-tensors are not affected. For dicts, the changes are done in-place!
    """
    if hasattr(x, 'to'):
        return x.to(device)
    if isinstance(x, list) or isinstance(x, tuple):
        return [move_to(i, device) for i in x]
    if isinstance(x, dict) or isinstance(x, defaultdict):
        for k, v in x.items():
            x[k] = move_to(v, device)
        return x
    return x


def find_lengths(messages: torch.Tensor) -> torch.Tensor:
    """
    :param messages: A tensor of term ids, encoded as Long values, of size (batch size, max sequence length).
    :returns A tensor with lengths of the sequences, including the end-of-sequence symbol <eos> (in EGG, it is 0).
    If no <eos> is found, the full length is returned (i.e. messages.size(1)).

    >>> messages = torch.tensor([[1, 1, 0, 0, 0, 1], [1, 1, 1, 10, 100500, 5]])
    >>> lengths = find_lengths(messages)
    >>> lengths
    tensor([3, 6])
    """
    max_k = messages.size(1)
    zero_mask = messages == 0
    # a bit involved logic, but it seems to be faster for large batches than slicing batch dimension and
    # querying torch.nonzero()
    # zero_mask contains ones on positions where 0 occur in the outputs, and 1 otherwise
    # zero_mask.cumsum(dim=1) would contain non-zeros on all positions after 0 occurred
    # zero_mask.cumsum(dim=1) > 0 would contain ones on all positions after 0 occurred
    # (zero_mask.cumsum(dim=1) > 0).sum(dim=1) equates to the number of steps that happened after 0 occured (including it)
    # max_k - (zero_mask.cumsum(dim=1) > 0).sum(dim=1) is the number of steps before 0 took place

    lengths = max_k - (zero_mask.cumsum(dim=1) > 0).sum(dim=1)
    lengths.add_(1).clamp_(max=max_k)

    return lengths


def dump_sender_receiver(game: torch.nn.Module,
                         dataset: 'torch.utils.data.DataLoader',
                         gs: bool, variable_length: bool,
                         device: Optional[torch.device] = None):
    """
    A tool to dump the interaction between Sender and Receiver
    :param game: A Game instance
    :param dataset: Dataset of inputs to be used when analyzing the communication
    :param gs: whether Gumbel-Softmax relaxation was used during training
    :param variable_length: whether variable-length communication is used
    :param device: device (e.g. 'cuda') to be used
    :return:
    """
    train_state = game.training  # persist so we restore it back
    game.eval()

    device = device if device is not None else device

    sender_inputs, messages, receiver_inputs, receiver_outputs = [], [], [], []
    labels = []

    with torch.no_grad():
        for batch in dataset:
            # by agreement, each batch is (sender_input, labels) plus optional (receiver_input)
            sender_input = move_to(batch[0], device)
            receiver_input = None if len(batch) == 2 else move_to(batch[2], device)

            message = game.sender(sender_input)

            # Under GS, the only output is a message; under Reinforce, two additional tensors are returned.
            # We don't need them.
            if not gs: message = message[0]

            output = game.receiver(message, receiver_input)
            if not gs: output = output[0]

            if batch[1] is not None:
                labels.extend(batch[1])

            if isinstance(sender_input, list) or isinstance(sender_input, tuple):
                sender_inputs.extend(zip(*sender_input))
            else:
                sender_inputs.extend(sender_input)

            if receiver_input is not None:
                receiver_inputs.extend(receiver_input)

            if gs: message = message.argmax(dim=-1)  # actual symbols instead of one-hot encoded

            if not variable_length:
                messages.extend(message)
                receiver_outputs.extend(output)
            else:
                # A trickier part is to handle EOS in the messages. It also might happen that not every message has EOS.
                # We cut messages at EOS if it is present or return the entire message otherwise. Note, EOS id is always
                # set to 0.

                for i in range(message.size(0)):
                    eos_positions = (message[i, :] == 0).nonzero()
                    message_end = eos_positions[0].item() if eos_positions.size(0) > 0 else -1
                    assert message_end == -1 or message[i, message_end] == 0
                    if message_end < 0:
                        messages.append(message[i, :])
                    else:
                        messages.append(message[i, :message_end + 1])

                    if gs:
                        receiver_outputs.append(output[i, message_end, ...])
                    else:
                        receiver_outputs.append(output[i, ...])

    game.train(mode=train_state)

    return sender_inputs, messages, receiver_inputs, receiver_outputs, labels
