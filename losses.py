"""Losses in main Equations 3–7. All pairs belong to one experimental history."""
import math
import torch
from torch.nn import functional as F


def pair_differences(scores, positions):
    scores = scores.flatten()
    i, j = torch.triu_indices(len(scores), len(scores), 1, device=scores.device)
    return scores[j] - scores[i], positions[j] - positions[i]


def ordering_losses(scores, positions, margin):
    """All-pair ranking, equal-count consistency and adjacent-state ranking.

    positions are exposure counts normalized over the training history, not
    renormalized over each sampled batch. Equal positions identify repeats.
    """
    difference, gap = pair_differences(scores, positions)
    ordered, equal = gap > 0, gap == 0
    zero = scores.sum() * 0
    ranking = (
        F.relu(margin * (0.25 + 0.75 * gap[ordered]) - difference[ordered]).mean()
        if ordered.any() else zero
    )
    repeat = difference[equal].abs().mean() if equal.any() else zero
    states = torch.unique(positions)
    means = torch.stack([scores.flatten()[positions == p].mean() for p in states])
    adjacent = (
        F.relu(margin * (0.25 + 0.75 * (states[1:] - states[:-1]))
               - (means[1:] - means[:-1])).mean()
        if len(states) > 1 else zero
    )
    return ranking, repeat, adjacent


def normal_losses(scores, reference_score):
    """Mean absolute deviation, reference deviation and population variance."""
    deviation = (scores - scores.mean()).abs().mean()
    reference = (scores - reference_score).abs().mean()
    variance = scores.var(unbiased=False)
    return deviation, reference, variance


def auxiliary_loss(scores, positions, temperature):
    difference, gap = pair_differences(scores, positions)
    valid = gap > 0
    return (F.softplus(-difference[valid] / temperature).mean()
            if valid.any() else scores.sum() * 0)


def confusion_loss(scores, positions, temperature):
    difference, gap = pair_differences(scores, positions)
    valid = gap > 0
    u = difference[valid] / temperature
    return ((0.5 * (F.softplus(u) + F.softplus(-u)) - math.log(2)).mean()
            if valid.any() else scores.sum() * 0)
