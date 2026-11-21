"""Verified source data for three UCI binary classification benchmarks.

Prepare a local data directory from the FusionBench repository root:

    python -m examples.uci_three_binary --download /tmp/uci3
    python -m examples.uci_three_binary /tmp/uci3

The second command verifies archive and member SHA-256, feature dimensions,
label counts, and finite feature values. populations(data_dir) performs the
same checks offline and yields (name, float64 features, int64 zero/one labels).
Original source row order and every feature column are preserved, including
the constant feature in Ionosphere. SOURCES records URLs, members, and label
encodings. No split, scaling, or imputation is applied.
"""

import argparse
import csv
import hashlib
import io
import tempfile
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Source:
    name: str
    url: str
    archive_sha256: str
    member: str
    member_sha256: str
    shape: tuple[int, int]
    labels: tuple[str, str]  # Raw labels for encoded 0 and 1, in that order.
    n_positive: int


SOURCES = (
    Source(
        "banknote",
        "https://archive.ics.uci.edu/static/public/267/banknote+authentication.zip",
        "1e2acd9a2085fadf3d8145c12d3d22af853320d52294a6590c2eaf75fdc05227",
        "data_banknote_authentication.txt",
        "d0539aaed2139ba7a587b3e34fb345ce503ff7d5d33dbf9912d8e195ce425cb9",
        (1372, 4),
        ("0", "1"),
        610,
    ),
    Source(
        "ionosphere",
        "https://archive.ics.uci.edu/static/public/52/ionosphere.zip",
        "4d218ece62756c99659011a13052d06464f13cb2b3d0410ce2aef16de1403860",
        "ionosphere.data",
        "46d52186b84e20be52918adb93e8fb9926b34795ff7504c24350ae0616a04bbd",
        (351, 34),
        ("b", "g"),
        225,
    ),
    Source(
        "spambase",
        "https://archive.ics.uci.edu/static/public/94/spambase.zip",
        "813ac1df8effac70463c09c9c4b11e8803eefcab54771af66150852bcdcd1636",
        "spambase.data",
        "b1ef93de71f97714d3d7d4f58fc9f718da7bbc8ac8a150eff2778616a8097b12",
        (4601, 57),
        ("0", "1"),
        1813,
    ),
)


def _verify(data, expected, name):
    actual = hashlib.sha256(data).hexdigest()
    if actual != expected:
        raise ValueError(
            f"SHA-256 mismatch for {name}: expected {expected}, received {actual}"
        )


def populations(data_dir):
    """Yield the verified datasets in banknote, ionosphere, spambase order."""
    for source in SOURCES:
        archive = Path(data_dir) / f"{source.name}.zip"
        content = archive.read_bytes()
        _verify(content, source.archive_sha256, archive.name)
        with zipfile.ZipFile(io.BytesIO(content)) as zipped:
            member = zipped.read(source.member)
        _verify(member, source.member_sha256, source.member)

        rows = list(csv.reader(io.StringIO(member.decode("ascii"))))
        n, d = source.shape
        if len(rows) != n or any(len(row) != d + 1 for row in rows):
            raise ValueError(
                f"unexpected data shape for {source.name}, expected {source.shape} plus labels"
            )
        x = np.array([row[:-1] for row in rows], dtype=np.float64)
        if not np.isfinite(x).all():
            raise ValueError(f"nonfinite features in {source.name}")
        labels = np.array([row[-1] for row in rows])
        if set(labels) != set(source.labels):
            raise ValueError(f"unexpected class labels in {source.name}")
        y = (labels == source.labels[1]).astype(np.int64)
        if y.sum() != source.n_positive:
            raise ValueError(f"unexpected class counts in {source.name}")
        yield source.name, x, y


def download(data_dir):
    """Populate an archive directory, verifying bytes before installing each file."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    for source in SOURCES:
        path = data_dir / f"{source.name}.zip"
        if path.exists():
            _verify(path.read_bytes(), source.archive_sha256, path.name)
            continue
        with urllib.request.urlopen(source.url, timeout=45) as response:
            content = response.read()
        _verify(content, source.archive_sha256, path.name)
        with tempfile.NamedTemporaryFile(dir=data_dir, delete=False) as tmp:
            temp_path = Path(tmp.name)
            try:
                tmp.write(content)
                tmp.close()
                temp_path.replace(path)
            finally:
                temp_path.unlink(missing_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    if args.download:
        download(args.data_dir)
    for name, x, y in populations(args.data_dir):
        print(
            f"{name}: {x.shape[0]} rows, {x.shape[1]} features, {y.sum()} positive labels"
        )
