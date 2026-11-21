"""Export the fixed design example and audit both frozen networks on all test states."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from fusionbench.eval.matching_panel import make_panel
from fusionbench.eval.matching_posterior import MatchingFamily


def export(out):
    out.mkdir(parents=True, exist_ok=True)
    low, high = 1 / 64, 63 / 64
    data = make_panel(4400, 6, "balanced")
    family = MatchingFamily(6)
    post = family.fit(
        data.fit.bits[:16],
        data.fit.y["matching"][:16],
        data.unlabeled.bits,
        pi=data.pi,
        low=low,
        high=high,
        mode="joint",
    )
    report = {
        "data_seed": 4400,
        "k": 6,
        "fit_labels": 16,
        "unlabeled_rows": 128,
        "train_design": "balanced",
        "fit_law": "matching",
        "mode": "joint",
        "input": "float64 [N,12], each provider score equals 1/64 (low) or 63/64 (high)",
        "output": "float64 [N,1], the linear readout before any clipping",
        "possible_matching_trees": len(post.weights),
        "networks": {},
    }
    for order in (1, 2):
        compiler = post.compile(order)
        model = compiler.to_torch()
        raw = torch.from_numpy(
            np.asarray(data.test[2].preds[:1, :, 1], dtype="<f8").copy()
        )
        path = out / f"matching_order{order}.pt"
        with torch.inference_mode():
            torch.jit.save(torch.jit.trace(model, raw), str(path))
            loaded = torch.jit.load(str(path), map_location="cpu")
            results = {}
            for size, block in sorted(data.test.items()):
                p = torch.from_numpy(
                    np.ascontiguousarray(block.preds[:, :, 1], dtype="<f8")
                )
                prediction = loaded(p).numpy().ravel()
                explicit = post.explicit(block.bits)
                compiled = compiler.predict(block.bits)
                max_eval = float(np.max(abs(prediction - compiled)))
                max_mix = float(np.max(abs(prediction - explicit)))
                np.testing.assert_allclose(prediction, compiled, atol=1e-12, rtol=0)
                if order == 2 or size <= 3:
                    np.testing.assert_allclose(prediction, explicit, atol=1e-12, rtol=0)
                results[size] = {
                    "states": len(p),
                    "max_abs_error_vs_compiled": max_eval,
                    "max_abs_error_vs_explicit": max_mix,
                    "max_abs_error_vs_explicit_after_clipping": float(
                        np.max(abs(np.clip(prediction, 0, 1) - explicit))
                    ),
                }
        report["networks"][order] = {
            "path": path.name,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "hidden_units": compiler.n_units,
            "nonzero_parameters": sum(
                int(torch.count_nonzero(p)) for p in model.parameters()
            ),
            "dense_parameters": sum(p.numel() for p in model.parameters()),
            "checks": results,
        }
    (out / "export.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--out", type=Path, required=True)
    args = cli.parse_args()
    torch.set_num_threads(1)
    export(args.out)
