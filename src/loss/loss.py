"""Loss functions for query embedding."""
from abc import abstractmethod

import torch
from torch import nn

__all__ = [
    "QueryEmbeddingLoss",
    "BCEQueryEmbeddingLoss",
    "LabelPredictionLoss",
    "softmax_with_cross_entropy",
]


class QueryEmbeddingLoss(nn.Module):
    """A loss for query embedding."""

    @abstractmethod
    def forward(
        self,
        scores: torch.FloatTensor,
        targets: torch.LongTensor,
    ) -> torch.FloatTensor:
        """
        Compute the loss for a batch of scores.

        .. note ::
            targets is assumed to contain unique elements.

        :param scores: shape: (batch_size, num_choices)
            The scores. Larger is better.
        :param targets: shape: (2, nnz)
            The targets.
        """
        raise NotImplementedError


class BCEQueryEmbeddingLoss(QueryEmbeddingLoss):
    """Binary cross-entropy loss without creating the dense target required by torch.nn.BCEWithLogits."""

    def forward(
        self,
        scores: torch.FloatTensor,
        targets: torch.LongTensor,
    ) -> torch.FloatTensor:  # noqa: D102
        batch_id, entity_id = targets

        # cf. https://github.com/pytorch/pytorch/blob/0d11dbf5119e1e4c69016ed1527bcf5013b208af/aten/src/ATen/native/Loss.cpp#L224-L245
        # auto max_val = (-input).clamp_min_(0);
        m = (-scores).clamp_min(0)

        # (1 - target).mul_(input).add_(max_val).add_((-max_val).exp_().add_((-input -max_val).exp_()).log_())
        # (1 - target) * input + max_val + log(exp(-max_val) + exp(-input - max_val))
        # input - target * input + max_val + log(exp(-max_val) + exp(-input - max_val))
        loss = scores.sum() - scores[batch_id, entity_id].sum() + m.sum() + torch.log(torch.exp(-m) + torch.exp(-scores - m)).sum()

        # apply_loss_reduction(loss, reduction)
        return loss / scores.numel()

class LabelPredictionLoss(nn.Module):
    """A loss for label prediction."""

    @abstractmethod
    def forward(
        self,
        scores: torch.FloatTensor,
        labels: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """
        Compute the loss for a batch of scores.

        .. note ::
            labels can be one-hot or soft labels.

        :param scores: shape: (batch_size, vocab_size)
            The scores. Larger is better.
        :param labels: shape: (batch_size, vocab_size)
            The labels.
        """
        raise NotImplementedError

class softmax_with_cross_entropy(LabelPredictionLoss):
    def __init__(self):
        super(softmax_with_cross_entropy,self).__init__()

    def forward(self, logits, label):
        logprobs=torch.nn.functional.log_softmax(logits,dim=1)
        loss=-1.0*torch.sum(torch.mul(label,logprobs),dim=1).squeeze()
        loss=torch.mean(loss)
        return loss