from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl
import wfdb
from tqdm import tqdm

from shdbaf.artifacts import segments_parquet, splits_parquet
from shdbaf.settings import load_settings


def normalize_aux_note(s: str) -> str | None:
    s = (s or "").strip()
    return s[1:] if s.startswith("(") else None


def main() -> None:
    S = load_settings()
    raw_dir: Path = S.paths.raw_path

    splits_path = splits_parquet()
    out_path = segments_parquet()

    if not splits_path.exists():
        raise FileNotFoundError(
            f"{splits_path} introuvable. Lance `pixi run split` d'abord."
        )
    if not raw_dir.exists():
        raise FileNotFoundError(
            f"raw_dir introuvable: {raw_dir} (lance `pixi run download`)."
        )

    splits = pl.read_parquet(splits_path).select(["record", "subject_id", "split"])
    items = list(splits.iter_rows())

    rows: list[dict[str, Any]] = []

    for rec, subject_id, split in tqdm(
        items, desc="Extracting rhythm segments", unit="record"
    ):
        base = raw_dir / rec  # sans extension

        hdr = wfdb.rdheader(str(base))
        fs = float(hdr.fs)
        sig_len = int(hdr.sig_len)

        ann = wfdb.rdann(str(base), "atr")
        samples = list(map(int, ann.sample))
        aux = list(ann.aux_note)

        # garder uniquement les marqueurs "(...)"
        markers = []
        for smp, note in zip(samples, aux):
            lab = normalize_aux_note(note)
            if lab is not None:
                markers.append((smp, lab))

        markers.sort(key=lambda x: x[0])
        if not markers:
            continue

        for i, (start, lab) in enumerate(markers):
            end = markers[i + 1][0] if i + 1 < len(markers) else sig_len
            if end <= start:
                continue

            rows.append(
                dict(
                    record=rec,
                    subject_id=subject_id,
                    split=split,
                    fs=fs,
                    sig_len=sig_len,
                    start_sample=int(start),
                    end_sample=int(end),
                    start_sec=float(start / fs),
                    end_sec=float(end / fs),
                    duration_sec=float((end - start) / fs),
                    label=str(lab),
                )
            )

    df = pl.DataFrame(rows)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out_path)

    print(f"✅ Wrote {out_path.resolve()} ({df.height} rhythm segments)")
    print("Labels (top 50):")
    print(df.group_by("label").len().sort("len", descending=True).head(50))


if __name__ == "__main__":
    main()
