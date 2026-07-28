import argparse
import urllib.request
import zipfile
from pathlib import Path

MEMERAG_ARCHIVE_URL = "https://github.com/amazon-science/MEMERAG/archive/refs/heads/main.zip"
DEFAULT_REPO_DIR = Path("data/MEMERAG")
NEEDED_SUBTREES = [("data", "memerag"), ("src", "prompts")]


def download_memerag(output_dir: Path = DEFAULT_REPO_DIR) -> Path:
    """Download MEMERAG's data and prompt templates into `output_dir`, unless already present.

    Parameters
    ----------
    output_dir : Path
        Directory to populate with a `data/memerag/` and `src/prompts/` subtree, matching the
        layout `load_memerag_dataset` and `load_prompt_templates` expect.

    Returns
    -------
    Path
        `output_dir`, for convenient chaining.
    """
    if (output_dir / "data" / "memerag" / "en.jsonl").exists():
        print(f"MEMERAG already present at {output_dir}, skipping download.")
        return output_dir

    output_dir.mkdir(parents=True, exist_ok=True)
    archive_path = output_dir.parent / (output_dir.name + ".zip")
    print(f"Downloading MEMERAG from {MEMERAG_ARCHIVE_URL} ...")
    urllib.request.urlretrieve(MEMERAG_ARCHIVE_URL, archive_path)

    print(f"Extracting data/memerag/ and src/prompts/ into {output_dir} ...")
    with zipfile.ZipFile(archive_path) as zf:
        for member in zf.namelist():
            parts = Path(member).parts
            if len(parts) <= 1:
                continue
            relative_path = Path(*parts[1:])  # strip the top-level "MEMERAG-main/" folder
            if not any(relative_path.parts[: len(subtree)] == subtree for subtree in NEEDED_SUBTREES):
                continue
            target = output_dir / relative_path
            if member.endswith("/"):
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(zf.read(member))

    archive_path.unlink()
    print(f"MEMERAG data and prompts saved to {output_dir}")
    return output_dir


def _parse_args():
    parser = argparse.ArgumentParser(description="Download MEMERAG's per-sentence data and judge prompt templates.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_REPO_DIR, help="Directory to download into.")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    download_memerag(args.output_dir)
