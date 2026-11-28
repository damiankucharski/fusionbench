"""Replay the frozen fixed-menu selection audit and development-fitted controls."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_limits

from fusionbench.eval.transfer_grid import brier
from fusionbench.eval.validation_curse import audit_menus
from fusionbench.methods.averaging import SimpleAverage, TopN
from fusionbench.methods.native_pattern import RegularizedNativePatternOKAPI
from fusionbench.methods.stacking import StackingLR

HERE = Path(__file__).resolve().parent
ROOT = 4_100_000
N = {"fit": 640, "validation": 512, "audit": 2048, "iid": 4096, "shift": 4096}
P = np.array([0.2, 0.4, 0.6, 0.8])
SOURCE_MASS = np.full(4, 0.25)
SHIFT_MASS = np.array([0.4, 0.1, 0.1, 0.4])
AMPLITUDE = 0.1


def signs(codes):
    return 1 - 2 * ((codes[:, None] >> np.arange(6)) & 1)


def providers(anchor, code):
    sign = signs(code)
    offsets = np.stack((sign, -sign), axis=2).reshape(len(code), 12)
    pos = anchor[:, None] + AMPLITUDE * offsets
    return np.stack((1 - pos, pos), axis=2)


# Entry (state, mask) is the product of the selected observable signs.
WALSH = np.array(
    [[(-1) ** ((c & m).bit_count()) for m in range(64)] for c in range(64)]
)


def candidates(x):
    anchor = x[:, :, 1].mean(axis=1)
    code = (x[:, ::2, 1] < x[:, 1::2, 1]) @ (2 ** np.arange(6))
    predictions = anchor[:, None] + AMPLITUDE * WALSH[code]
    predictions[:, 0] = anchor
    return predictions


def draw(seed, stream, n, shifted):
    rng = np.random.default_rng([seed, stream])
    p = rng.choice(P, n, p=SHIFT_MASS if shifted else SOURCE_MASS)
    code = rng.integers(0, 64, n)
    return providers(p, code), p, signs(code)[:, 0], rng.random(n)


def target(anchor, first_sign, uniform, positive, shifted):
    effect = (-1 if shifted else 1) * AMPLITUDE if positive else 0.0
    return (uniform < anchor + effect * first_sign).astype(int)


def population(positive, shifted):
    anchor = np.repeat(P, 64)
    codes = np.tile(np.arange(64), 4)
    eta = (
        anchor
        + (AMPLITUDE * (-1 if shifted else 1) if positive else 0) * signs(codes)[:, 0]
    )
    q = candidates(providers(anchor, codes))
    conditional = 2 * (eta[:, None] * (1 - q) ** 2 + (1 - eta[:, None]) * q**2)
    law = np.repeat((SHIFT_MASS if shifted else SOURCE_MASS) / 64, 64)
    risk = law @ conditional
    # The ten-point prediction grid, finite state space and given anchor masses
    # give exact risks with at most three decimal places. Restore equality of ties.
    np.testing.assert_allclose(risk, risk.round(3), atol=1e-13, rtol=0)
    return risk.round(3)


def write(path, rows):
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def digest(*arrays):
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def run(out):
    from loguru import logger

    logger.disable("okapi")
    start = perf_counter()
    raw = []
    refs = []
    banks = []
    riskrows = []
    risks = {}
    for name, positive in [("mean_null", False), ("information", True)]:
        source, shift = (population(positive, s) for s in (False, True))
        expected = (
            np.r_[0.400, np.full(63, 0.420)]
            if not positive
            else np.r_[0.400, 0.380, np.full(62, 0.420)]
        )
        np.testing.assert_allclose(source, expected, atol=1e-14, rtol=0)
        risks[name] = source, shift
        riskrows.extend(
            {"law": name, "candidate": i, "source_risk": u, "shift_risk": v}
            for i, (u, v) in enumerate(zip(source, shift, strict=True))
        )
    for rep in range(32):
        seed = ROOT + rep
        draws = {
            k: draw(seed, i, n, k == "shift") for i, (k, n) in enumerate(N.items())
        }
        q = {k: candidates(x) for k, (x, *_) in draws.items() if k != "fit"}
        for name, positive in [("mean_null", False), ("information", True)]:
            labels = {
                k: target(p, s, u, positive, k == "shift")
                for k, (_, p, s, u) in draws.items()
            }
            key = {"law": name, "replicate": rep, "seed": seed}
            for stream, (k, (x, _, _, u)) in enumerate(draws.items()):
                banks.append(
                    {
                        **key,
                        "split": k,
                        "stream": stream,
                        "n": len(x),
                        "providers": x.shape[1],
                        "features_sha256": digest(x),
                        "label_uniforms_sha256": digest(u),
                        "labels_sha256": digest(labels[k]),
                    }
                )
            losses = {k: 2 * np.square(z - labels[k][:, None]) for k, z in q.items()}
            rows = audit_menus(
                losses["validation"],
                losses["audit"],
                losses["iid"],
                losses["shift"],
                menu_sizes=(1, 4, 16, 64),
                validation_sizes=(32, 128, 512),
                population_risk=risks[name][0],
                shifted_risk=risks[name][1],
                tie_atol=1e-12,
            )
            raw.extend({**key, **r} for r in rows)
            methods = {
                "mean": SimpleAverage(),
                "top_n": TopN("brier"),
                "prob_lr": StackingLR(c_value=1),
            }
            if rep < 8:
                methods["native"] = RegularizedNativePatternOKAPI(
                    seed=seed, split_seed=seed
                )
            for label, model in methods.items():
                tick = perf_counter()
                model.fit(draws["fit"][0], labels["fit"], task="binary", n_classes=2)
                elapsed = perf_counter() - tick
                refs.append(
                    {
                        **key,
                        "method": label,
                        "fit_seconds": elapsed,
                        "native_choice": getattr(model, "chosen_", None),
                        "top_n": getattr(model, "n_selected_", None),
                        **{
                            k: brier(model.predict(x), labels[k])
                            for k, (x, *_) in draws.items()
                        },
                    }
                )
    out.mkdir(exist_ok=True, parents=True)
    for filename, table in [
        ("selections", raw),
        ("references", refs),
        ("banks", banks),
        ("population", riskrows),
    ]:
        write(out / (filename + ".csv"), table)
    (out / "run.json").write_text(
        json.dumps(
            {
                "base_seed": ROOT,
                "banks": 32,
                "native_banks_per_arm": 8,
                "rows": N,
                "menus": [1, 4, 16, 64],
                "validation_sizes": [32, 128, 512],
                "seconds": perf_counter() - start,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=HERE / "results")
    with threadpool_limits(1):
        run(parser.parse_args().out)
