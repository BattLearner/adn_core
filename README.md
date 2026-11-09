# ADN: Battery State-of-Safety Assessment

ADN learns a battery safety score from charging voltage and current. Overcharge
records provide relative-order supervision, while normal-cycling records are used
to constrain score variation and suppress aging information in the shared
features. The same architecture is trained separately for each battery type.

## Contents

```text
adn_core/
├── model.py          # Feature extractor, reference alignment and two heads
├── losses.py         # Ordering, normal-stability and adversarial losses
├── train.py          # Alternating optimization and training loop
├── config.json       # One example training configuration
├── requirements.txt
└── README.md
```

## Installation

Use Python 3.12 and install the dependencies:

```bash
python -m pip install -r requirements.txt
```

## Network and training

The shared extractor combines temporal convolutions, attention and a pooled
input-difference branch. Its response to the fixed reference is subtracted to
align the representation. The positive and negative parts of 32 signed
coordinates form 64 features. A primary head produces the safety-related score;
an auxiliary head learns normal-cycle order during training.

## Input format

The training loop receives an iterable of dictionaries containing PyTorch
tensors. One dictionary is used per update:

| Key                | Shape          | Meaning                                                              |
| ------------------ | -------------- | -------------------------------------------------------------------- |
| `abuse`          | `[Na, L, 4]` | Charging inputs from one abuse history                               |
| `abuse_order`    | `[Na]`       | Normalized prior-overcharge counts                                   |
| `normal`         | `[Nn, L, 4]` | Charging inputs from one normal history                              |
| `normal_order`   | `[Nn]`       | Normalized normal-cycle positions                                    |
| `complete_abuse` | `[Na, L, 4]` | Matching complete-window inputs, supplied for partial-window updates |

`Na` and `Nn` are the numbers of observations; `L` is the number of sampled
positions, conventionally 256.

**To improve code readability, the code structure has been reorganized with the assistance of AI. All changes have been thoroughly reviewed and verified by the author to ensure consistent functionality and execution behavior.**
