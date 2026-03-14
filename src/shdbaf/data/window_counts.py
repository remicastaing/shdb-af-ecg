from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import polars as pl

from shdbaf.artifacts import segments_parquet, splits_parquet
from shdbaf.settings import load_settings


@dataclass(frozen=True)
class WindowCountConfig:
    win_sec: float
    stride_sec: float | None
    majority_thr: float
    drop_labels: Tuple[str, ...] = ("AB",)  # optionnel (diagnostic)

    def stride(self) -> float:
        return self.win_sec if self.stride_sec is None else self.stride_sec


def _label_map_svt(lab: str) -> str | None:
    # 5 classes: N, AFIB, AFL, AT, SVT (=NOD+PAT)
    if lab in ("N", "AFIB", "AFL", "AT"):
        return lab
    if lab in ("NOD", "PAT"):
        return "SVT"
    return None


def _label_map_other(lab: str) -> str | None:
    # 5 classes: N, AFIB, AFL, AT, OTHER (tout le reste)
    if lab in ("N", "AFIB", "AFL", "AT"):
        return lab
    return "OTHER"


def _label_map_6(lab: str) -> str | None:
    # 6 classes strictes
    if lab in ("N", "AFIB", "AFL", "AT", "NOD", "PAT"):
        return lab
    return None


def _count_windows_for_record(
    segs: pl.DataFrame,
    win_sec: float,
    stride_sec: float,
    majority_thr: float,
) -> Tuple[Dict[str, int], int, int]:
    rec_end = float(segs["end_sec"].max())
    n_kept = 0
    n_skipped = 0
    counts: Dict[str, int] = {}

    t = 0.0
    t_end = rec_end - win_sec
    if t_end < 0:
        return counts, 0, 0

    starts = segs["start_sec"].to_list()
    ends = segs["end_sec"].to_list()
    labs = segs["label_mapped"].to_list()

    j = 0
    nseg = len(starts)

    while t <= t_end + 1e-9:
        w0, w1 = t, t + win_sec

        while j < nseg and ends[j] <= w0:
            j += 1

        dur: Dict[str, float] = {}
        k = j
        while k < nseg and starts[k] < w1:
            a = max(w0, starts[k])
            b = min(w1, ends[k])
            if b > a:
                lab = labs[k]
                dur[lab] = dur.get(lab, 0.0) + (b - a)
            k += 1

        if not dur:
            n_skipped += 1
            t += stride_sec
            continue

        best_lab, best_d = max(dur.items(), key=lambda x: x[1])
        if best_d / win_sec >= majority_thr:
            counts[best_lab] = counts.get(best_lab, 0) + 1
            n_kept += 1
        else:
            n_skipped += 1

        t += stride_sec

    return counts, n_kept, n_skipped


def estimate_counts(label_scheme: str, cfg: WindowCountConfig) -> None:
    seg_path = segments_parquet()
    split_path = splits_parquet()

    if not seg_path.exists():
        raise FileNotFoundError(
            f"{seg_path} introuvable. Lance d'abord `pixi run segments`."
        )
    if not split_path.exists():
        raise FileNotFoundError(
            f"{split_path} introuvable. Lance d'abord `pixi run split`."
        )

    stride = cfg.stride()

    seg = pl.read_parquet(seg_path)

    # drop labels (diagnostic)
    if cfg.drop_labels:
        seg = seg.filter(~pl.col("label").is_in(list(cfg.drop_labels)))

    if label_scheme == "svt":
        mapper = _label_map_svt
        scheme_name = "5 classes: N/AFIB/AFL/AT/SVT(NOD+PAT)"
    elif label_scheme == "other":
        mapper = _label_map_other
        scheme_name = "5 classes: N/AFIB/AFL/AT/OTHER"
    elif label_scheme == "6":
        mapper = _label_map_6
        scheme_name = "6 classes: N/AFIB/AFL/AT/NOD/PAT"
    else:
        raise ValueError("label_scheme must be one of: svt, other, 6")

    seg = seg.with_columns(
        pl.col("label").map_elements(mapper, return_dtype=pl.Utf8).alias("label_mapped")
    ).filter(pl.col("label_mapped").is_not_null())

    total_counts: Dict[str, int] = {}
    per_split_counts: Dict[str, Dict[str, int]] = {"train": {}, "val": {}, "test": {}}
    kept_total = 0
    skipped_total = 0

    for (rec, split), g in seg.group_by(["record", "split"], maintain_order=True):
        c, kept, skipped = _count_windows_for_record(
            g.select(["start_sec", "end_sec", "label_mapped"]),
            win_sec=cfg.win_sec,
            stride_sec=stride,
            majority_thr=cfg.majority_thr,
        )
        kept_total += kept
        skipped_total += skipped

        for lab, n in c.items():
            total_counts[lab] = total_counts.get(lab, 0) + n
            d = per_split_counts.get(split, {})
            d[lab] = d.get(lab, 0) + n
            per_split_counts[split] = d

    print("\n" + "=" * 80)
    print(f"Label scheme: {scheme_name}")
    print(
        f"win_sec={cfg.win_sec}, stride_sec={stride}, majority_thr={cfg.majority_thr}, drop={cfg.drop_labels}"
    )
    print(f"kept_windows={kept_total}, skipped/mixed={skipped_total}")

    def _sorted(d: Dict[str, int]) -> List[Tuple[str, int]]:
        return sorted(d.items(), key=lambda x: x[1], reverse=True)

    print("\nGlobal counts:")
    for lab, n in _sorted(total_counts):
        print(f"  {lab:>6s} : {n}")

    print("\nBy split:")
    for sp in ("train", "val", "test"):
        print(f"  [{sp}]")
        for lab, n in _sorted(per_split_counts.get(sp, {})):
            print(f"    {lab:>6s} : {n}")


def main() -> None:
    S = load_settings()

    cfg = WindowCountConfig(
        win_sec=float(S.dataset.win_sec),
        stride_sec=float(S.dataset.stride_sec)
        if S.dataset.stride_sec is not None
        else None,
        majority_thr=float(S.dataset.majority_thr),
        drop_labels=(
            "AB",
        ),  # diagnostic (tu peux le mettre dans settings.toml plus tard)
    )

    for scheme in ("svt", "other", "6"):
        estimate_counts(scheme, cfg)


if __name__ == "__main__":
    main()
