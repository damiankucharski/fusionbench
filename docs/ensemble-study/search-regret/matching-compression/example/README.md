# Run a compiled matching ensemble

These frozen CPU networks accept twelve provider probabilities per example. Each input must equal 1/64 or 63/64. In this example the joint posterior was fitted to 16 labels and 128 additional unlabeled inputs from the balanced design, using bank 4400. The original ensemble has 10,395 possible matching trees.

```python
import torch

network = torch.jit.load("matching_order2.pt", map_location="cpu").eval()
scores = torch.full((1, 12), 1 / 64, dtype=torch.float64)
scores[0, [0, 2, 5, 8]] = 63 / 64
with torch.inference_mode():
    probability = network(scores)[0, 0].item()
```

The order-one network uses 66 ReLU units and agrees with the ensemble when at most three inputs are high. The order-two network uses 561 units, adds the quadruple correction and agrees through five high inputs. Each model returns its raw scalar readout; callers can observe values outside the probability interval beyond that model's supported cardinality. These files start from provider scores and require only PyTorch at inference. [Exhaustive two-, three- and four-input checks and file hashes](export.json) · [Rebuild both files](../export_example.py)
