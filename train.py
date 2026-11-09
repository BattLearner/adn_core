"""ADN training on caller-supplied tensors."""
import math

import torch
from torch.nn import functional as F

from losses import ordering_losses, normal_losses, auxiliary_loss, confusion_loss

# Example coefficients for the full objective; callers can supply other weights.
DEFAULT_WEIGHTS = {
    "normal_deviation": 1.0, "reference_deviation": 2.0,
    "normal_variance": 0.2, "feature_confusion": 0.01,
    "equal_count": 0.1, "adjacent_order": 0.8,
    "segment_consistency": 0.3, "feature_magnitude": 1e-5,
}


def make_optimizers(model, learning_rate):
    main = [p for name, p in model.named_parameters() if not name.startswith("g_C.")]
    primary = torch.optim.AdamW(main, lr=learning_rate, weight_decay=1e-4)
    auxiliary = torch.optim.AdamW(model.g_C.parameters(), lr=learning_rate,
                                 weight_decay=0.001)
    return primary, auxiliary


def training_step(model, main_optimizer, auxiliary_optimizer,
                  abuse, abuse_order, normal, normal_order, *,
                  step, schedule_steps, base_lr, weights=None, margin=0.5,
                  temperature=1.0, ramp_steps=200, partial_window=False,
                  complete_abuse=None, augmentation_factor=None):
    """Run auxiliary updates, then one extractor/primary-head update.

    Each domain batch must come from a single history, ordered by exposure or
    cycle position. abuse/normal: [observations, positions, 4], with channels
    [observation voltage, observation current, reference voltage, reference current].
    For a partial window, complete_abuse contains matching full-window records.
    The caller supplies the histories and charging windows for each update.
    """
    weights = DEFAULT_WEIGHTS if weights is None else weights
    if not 1 <= step <= schedule_steps:
        raise ValueError("step must lie within the learning-rate schedule")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    for x, p in [(abuse, abuse_order), (normal, normal_order)]:
        if x.ndim != 3 or x.shape[-1] != 4 or len(x) == 0 or p.ndim != 1 or len(x) != len(p):
            raise ValueError("Bad batch/position shape")
        if not torch.isfinite(x).all() or not torch.isfinite(p).all() or (p[1:] < p[:-1]).any():
            raise ValueError("Inputs must be finite and positions ordered")
    if partial_window and (complete_abuse is None or complete_abuse.shape != abuse.shape):
        raise ValueError("Partial-window consistency requires matched complete windows")
    model.train()
    if augmentation_factor is None:
        augmentation_factor = float(torch.empty((), device=normal.device).uniform_(0.8, 1.8))
    augmented = normal.clone()
    augmented[:, :, :2] = normal[:, :, 2:] + augmentation_factor * (normal[:, :, :2] - normal[:, :, 2:])

    # Feature-level objective: the extractor is fixed during auxiliary updates.
    with torch.no_grad():
        _, normal_features = model(augmented)
        model.update_normal_scale(normal_features)
    auxiliary_parameters = list(model.g_C.parameters())
    for _ in range(2 if weights["feature_confusion"] > 0 else 0):
        auxiliary_optimizer.zero_grad(set_to_none=True)
        auxiliary_scores = model.g_C(model.probe_features(normal_features.detach()))
        loss_aux = auxiliary_loss(auxiliary_scores, normal_order, temperature)
        if (normal_order[1:] > normal_order[:-1]).any():
            loss_aux.backward()
            torch.nn.utils.clip_grad_norm_(auxiliary_parameters, 5.0)
            auxiliary_optimizer.step()
    auxiliary_optimizer.zero_grad(set_to_none=True)

    # Task-level losses and symmetric feature confusion update the main network.
    main_optimizer.zero_grad(set_to_none=True)
    score_abuse, feature_abuse = model(abuse)
    score_normal, feature_normal = model(augmented)
    reference = augmented[:1].clone()
    reference[:, :, :2] = reference[:, :, 2:]
    score_reference, _ = model(reference)
    rank, repeat, adjacent = ordering_losses(score_abuse, abuse_order, margin)
    deviation, reference_deviation, variance = normal_losses(score_normal, score_reference)
    fixed_head_scores = F.linear(model.probe_features(feature_normal), model.g_C.weight.detach())
    confusion = confusion_loss(fixed_head_scores, normal_order, temperature)
    consistency = score_abuse.sum() * 0
    if partial_window and weights["segment_consistency"] > 0:
        with torch.no_grad():
            target, _ = model(complete_abuse)
        consistency = F.smooth_l1_loss(score_abuse, target, beta=0.2)
    regularization = feature_abuse.square().mean() + feature_normal.square().mean()
    ramp = min(1.0, step / max(ramp_steps, 1))
    normal_objective = (weights["normal_deviation"] * deviation
                        + weights["reference_deviation"] * reference_deviation
                        + weights["normal_variance"] * variance)
    total = (rank + weights["equal_count"] * repeat
             + weights["adjacent_order"] * adjacent
             + ramp * (normal_objective + weights["feature_confusion"] * confusion)
             + weights["segment_consistency"] * consistency
             + weights["feature_magnitude"] * regularization)
    if not torch.isfinite(total):
        raise FloatingPointError("Non-finite training loss")
    total.backward()
    main_parameters = [p for name, p in model.named_parameters() if not name.startswith("g_C.")]
    norm = torch.nn.utils.clip_grad_norm_(main_parameters, 5.0)
    if not torch.isfinite(norm):
        raise FloatingPointError("Non-finite gradient")
    lr = base_lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (step - 1) / max(schedule_steps - 1, 1))))
    for optimizer in (main_optimizer, auxiliary_optimizer):
        for group in optimizer.param_groups:
            group["lr"] = lr
    main_optimizer.step()
    values = dict(total=total, ordering=rank, normal_deviation=deviation,
                  reference_deviation=reference_deviation, normal_variance=variance,
                  feature_confusion=confusion, equal_count=repeat,
                  adjacent_order=adjacent, segment_consistency=consistency,
                  feature_magnitude=regularization)
    return {name: float(value.detach()) for name, value in values.items()}


def training_calibration(reference_scores, abuse_scores):
    """Use training scores from complete diagnostic windows only."""
    low = float(reference_scores.detach().min())
    high = float(abuse_scores.detach().max())
    if not math.isfinite(low + high) or high <= low:
        raise ValueError("Training scores do not define a positive calibration span")
    return low, high


def score_to_sos(score, low, high):
    """Main Equation 8; preserve values outside the training-derived endpoints."""
    if high <= low:
        raise ValueError("Calibration span must be positive")
    return 100 * (high - score) / (high - low)


def train(model, batches, config):
    """Fit a model using an iterable of tensor batches; return the loss history.

    Each batch contains abuse, abuse_order, normal and normal_order. For a
    partial charging window, also provide the matching complete_abuse tensor.
    The model is updated in place. File loading is handled by the caller.
    """
    steps = config["steps"]
    schedule_steps = config["schedule_steps"]
    if not isinstance(steps, int) or steps <= 0:
        raise ValueError("steps must be a positive integer")
    if not isinstance(schedule_steps, int) or schedule_steps < steps:
        raise ValueError("schedule_steps must be an integer at least as large as steps")
    for name in ("learning_rate", "margin", "feature_temperature"):
        if not math.isfinite(config[name]) or config[name] <= 0:
            raise ValueError(f"{name} must be positive and finite")
    weights = DEFAULT_WEIGHTS | config["loss_weights"]
    if set(weights) != set(DEFAULT_WEIGHTS):
        raise ValueError("Unknown loss weight in config")
    if any(not math.isfinite(v) or v < 0 for v in weights.values()):
        raise ValueError("Loss weights must be finite and nonnegative")

    main_optimizer, auxiliary_optimizer = make_optimizers(model, config["learning_rate"])
    batches = iter(batches)
    history = []
    for step in range(1, steps + 1):
        try:
            batch = next(batches)
        except StopIteration:
            raise ValueError(f"Expected {steps} training batches; received {step - 1}") from None
        complete = batch.get("complete_abuse")
        losses = training_step(
            model, main_optimizer, auxiliary_optimizer,
            batch["abuse"], batch["abuse_order"],
            batch["normal"], batch["normal_order"],
            step=step, schedule_steps=schedule_steps,
            base_lr=config["learning_rate"], weights=weights,
            margin=config["margin"], temperature=config["feature_temperature"],
            partial_window=complete is not None, complete_abuse=complete,
        )
        history.append({"step": step, **losses})
    return history
