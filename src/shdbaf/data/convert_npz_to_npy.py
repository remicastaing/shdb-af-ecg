from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", type=str, required=True)
    ap.add_argument("--out-dir", type=str, required=True)
    ap.add_argument("--copy-manifest", action="store_true", default=True)
    ap.add_argument("--overwrite", action="store_true", default=False)
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    npz_files = sorted(in_dir.glob("chunk_*.npz"))
    if not npz_files:
        raise FileNotFoundError(f"Aucun chunk_*.npz trouvé dans {in_dir.resolve()}")

    n_chunks = 0
    n_windows = 0
    shapes = set()

    for f in tqdm(npz_files, desc="Convert NPZ -> NPY(memmap)", unit="chunk"):
        z = np.load(f, allow_pickle=False)

        if "x" not in z or "y" not in z:
            raise KeyError(f"{f.name} ne contient pas 'x' et 'y'")

        x = z["x"]  # (B, T, C)
        y = z["y"]  # (B,)

        stem = f.stem  # chunk_00081
        x_out = out_dir / f"{stem}_x.npy"
        y_out = out_dir / f"{stem}_y.npy"

        if (x_out.exists() or y_out.exists()) and not args.overwrite:
            # On considère que c'est déjà converti
            n_chunks += 1
            n_windows += int(x.shape[0])
            shapes.add(tuple(x.shape[1:]))
            continue

        np.save(x_out, x.astype(np.float32, copy=False))
        np.save(y_out, y.astype(np.int64, copy=False))

        n_chunks += 1
        n_windows += int(x.shape[0])
        shapes.add(tuple(x.shape[1:]))

    # Copier manifest (il référence encore chunk_XXXXX.npz, mais notre Dataset memmap
    # utilise Path(chunk).stem pour retrouver chunk_XXXXX_x.npy / y.npy, donc c'est OK)
    if args.copy_manifest:
        m = in_dir / "manifest.parquet"
        if m.exists():
            dest = out_dir / "manifest.parquet"
            if args.overwrite or not dest.exists():
                dest.write_bytes(m.read_bytes())

    report = {
        "in_dir": str(in_dir.resolve()),
        "out_dir": str(out_dir.resolve()),
        "chunks": n_chunks,
        "windows": n_windows,
        "x_inner_shapes": sorted(list(shapes)),
        "dtype_x": "float32",
        "dtype_y": "int64",
    }
    rep_path = out_dir / "conversion_report.json"
    rep_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"✅ Converted {n_chunks} chunks -> {out_dir.resolve()}")
    print(f"   windows: {n_windows}")
    print(f"   shapes(T,C): {report['x_inner_shapes']}")
    print(f"   report: {rep_path.name}")


if __name__ == "__main__":
    main()
