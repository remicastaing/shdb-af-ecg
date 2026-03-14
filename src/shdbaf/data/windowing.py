from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import polars as pl
import wfdb
from tqdm import tqdm

from shdbaf.settings import load_settings
from shdbaf.artifacts import (
    splits_parquet,
    segments_parquet,
    processed_dataset_dir,
    manifest_parquet,
)


def _select_channels(x: np.ndarray, channels: str) -> np.ndarray:
    if channels == "both":
        return x
    if channels == "ecg1":
        return x[:, [0]]
    if channels == "ecg2":
        return x[:, [1]] if x.shape[1] > 1 else x[:, [0]]
    raise ValueError("channels must be: ecg1, ecg2, both")


def _label_to_int(classes: Tuple[str, ...]) -> Dict[str, int]:
    return {c: i for i, c in enumerate(classes)}


def _majority_label_for_window(
    seg_starts: np.ndarray,
    seg_ends: np.ndarray,
    seg_labels: np.ndarray,
    w0: float,
    w1: float,
) -> Tuple[int | None, float]:
    mask = (seg_starts < w1) & (seg_ends > w0)
    if not mask.any():
        return None, 0.0

    starts = np.maximum(seg_starts[mask], w0)
    ends = np.minimum(seg_ends[mask], w1)
    labs = seg_labels[mask]
    durs = np.maximum(0.0, ends - starts)

    dur_by_label: Dict[int, float] = {}
    for lab, dur in zip(labs, durs):
        dur_by_label[int(lab)] = dur_by_label.get(int(lab), 0.0) + float(dur)

    best_lab = max(dur_by_label.items(), key=lambda x: x[1])[0]
    best_dur = dur_by_label[best_lab]
    frac = best_dur / (w1 - w0)
    return best_lab, frac


def _iter_windows(duration_sec: float, win_sec: float, stride_sec: float):
    t_end = duration_sec - win_sec
    if t_end < 0:
        return
    t = 0.0
    while t <= t_end + 1e-9:
        yield (t, t + win_sec)
        t += stride_sec


def main() -> None:
    S = load_settings()

    raw_dir = S.paths.raw_path
    split_path = splits_parquet()
    seg_path = segments_parquet()

    if not split_path.exists():
        raise FileNotFoundError(f"{split_path} introuvable. Lance `pixi run split`.")
    if not seg_path.exists():
        raise FileNotFoundError(f"{seg_path} introuvable. Lance `pixi run segments`.")

    # --- paramètres dataset depuis settings.toml
    win_sec = float(S.dataset.win_sec)
    stride_sec = float(S.dataset.stride_sec)
    majority_thr = float(S.dataset.majority_thr)
    channels = str(S.dataset.channels)
    classes = tuple(S.dataset.classes)
    chunk_size = int(S.dataset.chunk_size)

    if S.dataset.format != "npy":
        raise RuntimeError(
            f"settings.toml demande dataset.format={S.dataset.format!r}, "
            "mais ce pipeline windowing.py est volontairement NPY-only (perf training)."
        )

    tag = S.dataset.tag()
    out_dir = processed_dataset_dir(tag)
    out_dir.mkdir(parents=True, exist_ok=True)

    splits = pl.read_parquet(split_path).select(["record", "subject_id", "split"])
    seg = pl.read_parquet(seg_path).filter(pl.col("label").is_in(list(classes)))
    lab2i = _label_to_int(classes)

    # buffers chunk
    X_buf: List[np.ndarray] = []
    y_buf: List[int] = []
    rec_buf: List[str] = []
    split_buf: List[str] = []
    start_samp_buf: List[int] = []
    manifest_rows: list[dict[str, Any]] = []

    chunk_id = 0
    offset_in_chunk = 0

    def flush_chunk() -> None:
        nonlocal chunk_id, offset_in_chunk, X_buf, y_buf, rec_buf, split_buf, start_samp_buf
        if not X_buf:
            return

        chunk_id += 1
        stem = f"chunk_{chunk_id:05d}"

        x = np.stack(X_buf, axis=0).astype(np.float32)  # (B, T, C)
        y = np.array(y_buf, dtype=np.int64)

        # --- NPY memmap-friendly
        np.save(out_dir / f"{stem}_x.npy", x)
        np.save(out_dir / f"{stem}_y.npy", y)

        # meta (optionnel mais pratique)
        np.save(out_dir / f"{stem}_record.npy", np.array(rec_buf, dtype="U"))
        np.save(out_dir / f"{stem}_split.npy", np.array(split_buf, dtype="U"))
        np.save(out_dir / f"{stem}_start_sample.npy", np.array(start_samp_buf, dtype=np.int64))

        X_buf, y_buf, rec_buf, split_buf, start_samp_buf = [], [], [], [], []
        offset_in_chunk = 0

    for rec, subject_id, split in tqdm(list(splits.iter_rows()), desc="Windowing", unit="record"):
        srec = seg.filter(pl.col("record") == rec).select(["start_sec", "end_sec", "label"])
        if srec.height == 0:
            continue

        hdr = wfdb.rdheader(str(raw_dir / rec))
        fs = float(hdr.fs)
        sig_len = int(hdr.sig_len)
        duration_sec = sig_len / fs

        rr = wfdb.rdrecord(str(raw_dir / rec))
        x_full = _select_channels(rr.p_signal, channels)

        seg_starts = np.array(srec["start_sec"].to_list(), dtype=np.float64)
        seg_ends = np.array(srec["end_sec"].to_list(), dtype=np.float64)
        seg_labels = np.array([lab2i[l] for l in srec["label"].to_list()], dtype=np.int64)

        for (w0, w1) in _iter_windows(duration_sec, win_sec, stride_sec):
            lab, frac = _majority_label_for_window(seg_starts, seg_ends, seg_labels, w0, w1)
            if lab is None or frac < majority_thr:
                continue

            s0 = int(round(w0 * fs))
            s1 = int(round(w1 * fs))
            if s1 > x_full.shape[0]:
                continue

            X_buf.append(x_full[s0:s1, :])
            y_buf.append(int(lab))
            rec_buf.append(rec)
            split_buf.append(split)
            start_samp_buf.append(s0)

            manifest_rows.append(
                {
                    "record": rec,
                    "subject_id": subject_id,
                    "split": split,
                    "win_sec": win_sec,
                    "stride_sec": stride_sec,
                    "fs": fs,
                    "start_sec": w0,
                    "end_sec": w1,
                    "start_sample": s0,
                    "label": classes[int(lab)],
                    "label_id": int(lab),
                    "chunk": f"chunk_{chunk_id + 1:05d}",
                    "offset": offset_in_chunk,
                }
            )
            offset_in_chunk += 1

            if len(X_buf) >= chunk_size:
                flush_chunk()

    flush_chunk()

    manifest = pl.DataFrame(manifest_rows)
    m_path = manifest_parquet(tag)
    manifest.write_parquet(m_path)

    print(f"✅ Wrote {m_path.resolve()} ({manifest.height} windows)")
    print(manifest.group_by(["split", "label"]).len().sort(["split", "len"], descending=[False, True]))
    print(f"Dataset tag: {tag}")
    print(f"Output dir : {out_dir.resolve()}")


if __name__ == "__main__":
    main()