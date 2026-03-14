from __future__ import annotations

from pathlib import Path
import wfdb

from shdbaf.settings import load_settings


DB = "shdb-af"  # slug PhysioNet


def dl_required(dl_dir: Path, filename: str) -> None:
    """Télécharge un fichier qui DOIT exister. Si échec => erreur explicite."""
    wfdb.dl_files(
        db=DB,
        dl_dir=str(dl_dir),
        files=[filename],
        keep_subdirs=False,
        overwrite=False,
    )
    local = dl_dir / filename
    if not local.exists():
        raise RuntimeError(f"Fichier requis non téléchargé: {local}")


def dl_optional(dl_dir: Path, filename: str) -> bool:
    """Télécharge un fichier optionnel (ex: .atr). Retourne True si OK."""
    try:
        wfdb.dl_files(
            db=DB,
            dl_dir=str(dl_dir),
            files=[filename],
            keep_subdirs=False,
            overwrite=False,
        )
        return (dl_dir / filename).exists()
    except Exception as e:
        msg = str(e)
        if "404" in msg or "Not Found" in msg:
            return False
        raise


def main() -> None:
    S = load_settings()

    # IMPORTANT:
    # On suppose que settings.toml contient raw_dir="data/raw/shdb-af"
    # donc dl_dir est directement le dossier cible des fichiers WFDB.
    dl_dir = S.paths.raw_path
    dl_dir.mkdir(parents=True, exist_ok=True)

    # 1) Fichiers racine indispensables
    dl_required(dl_dir, "RECORDS.txt")
    dl_optional(dl_dir, "README.md")
    dl_optional(dl_dir, "AdditionalData.csv")
    dl_optional(dl_dir, "SHA256SUMS.txt")

    # 2) Liste des enregistrements
    records = [l.strip() for l in (dl_dir / "RECORDS.txt").read_text().splitlines() if l.strip()]

    # 3) Téléchargement des WFDB records
    for i, rec in enumerate(records, start=1):
        # requis
        dl_required(dl_dir, f"{rec}.hea")
        dl_required(dl_dir, f"{rec}.dat")
        # optionnels
        dl_optional(dl_dir, f"{rec}.atr")
        dl_optional(dl_dir, f"{rec}.qrs")

        if i % 10 == 0:
            print(f"... {i}/{len(records)} records")

    print(f"✅ Download OK: {dl_dir.resolve()}")
    print(f"   records: {len(records)}")
    ex = sorted([p.name for p in dl_dir.glob('*.hea')])[:5]
    print(f"   exemples: {ex}")


if __name__ == "__main__":
    main()