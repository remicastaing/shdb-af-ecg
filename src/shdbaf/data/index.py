from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl
import wfdb

from shdbaf.settings import load_settings
from shdbaf.artifacts import index_parquet


def _list_records_from_headers(raw_dir: Path) -> list[Path]:
    """Retourne la liste des headers (.hea) présents localement."""
    return sorted(raw_dir.glob("*.hea"))


def _safe_rdheader(base_path_no_ext: str) -> wfdb.Record:
    """
    Lit uniquement l'entête WFDB (sans charger le .dat).
    base_path_no_ext: chemin sans extension, ex '.../data/raw/shdb-af/010'
    """
    return wfdb.rdheader(base_path_no_ext)


def build_index(raw_dir: Path) -> pl.DataFrame:
    hea_files = _list_records_from_headers(raw_dir)
    if not hea_files:
        raise FileNotFoundError(f"Aucun .hea trouvé dans {raw_dir.resolve()}")

    rows: list[dict[str, Any]] = []
    for hea in hea_files:
        rec_id = hea.stem
        base = hea.with_suffix("")  # chemin sans extension
        base_str = str(base)

        dat_ok = base.with_suffix(".dat").exists()
        atr_ok = base.with_suffix(".atr").exists()
        qrs_ok = base.with_suffix(".qrs").exists()

        h = _safe_rdheader(base_str)

        fs = float(h.fs)
        n_sig = int(h.n_sig)
        sig_names = list(getattr(h, "sig_name", []))
        sig_len = int(getattr(h, "sig_len", 0))
        duration_sec = float(sig_len / fs) if fs > 0 and sig_len > 0 else None

        rows.append(
            {
                "record": rec_id,
                "path": str(base.relative_to(raw_dir)),  # ex: '010'
                "fs": fs,
                "n_sig": n_sig,
                "sig_names": sig_names,
                "sig_len": sig_len,
                "duration_sec": duration_sec,
                "has_dat": dat_ok,
                "has_atr": atr_ok,
                "has_qrs": qrs_ok,
                "base_date": getattr(h, "base_date", None),
                "base_time": getattr(h, "base_time", None),
            }
        )

    df = pl.DataFrame(rows)

    missing_dat = df.filter(~pl.col("has_dat")).height
    if missing_dat:
        print(f"⚠️  Attention: {missing_dat} records ont un .hea mais pas de .dat")

    return df.sort("record")


def main() -> None:
    S = load_settings()

    raw_dir = S.paths.raw_path
    out_path = index_parquet()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = build_index(raw_dir=raw_dir)
    df.write_parquet(out_path)

    print(f"✅ Index écrit: {out_path.resolve()} ({df.height} records)")
    counts = (
        df.select(pl.col("has_atr").cast(pl.Int8).alias("annotated"))
        .group_by("annotated")
        .count()
        .sort("annotated")
    )
    print("Résumé annotated (0=non,1=oui):")
    print(counts)


if __name__ == "__main__":
    main()
