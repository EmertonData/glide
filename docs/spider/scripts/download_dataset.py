import argparse
import json
import re
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

DEFAULT_GDRIVE_FILE_ID = "1403EGqzIDoHMdQF4c9Bkyl7dZLZ5Wt6J"
DEFAULT_OUTPUT_DIR = Path("data/spider")


def _resolve_gdrive_url(file_id: str) -> str:
    warning_url = f"https://drive.google.com/uc?export=download&id={file_id}"
    req = urllib.request.Request(warning_url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as response:
        content_type = response.headers.get("Content-Type", "")
        if "text/html" not in content_type:
            return response.url
        html = response.read().decode("utf-8")

    uuid_match = re.search(r'name="uuid"\s+value="([^"]+)"', html)
    if uuid_match is None:
        raise RuntimeError("Could not find UUID in Google Drive warning page. The sharing link may have changed.")

    uuid = uuid_match.group(1)
    params = {"id": file_id, "export": "download", "confirm": "t", "uuid": uuid}
    download_url = f"https://drive.usercontent.google.com/download?{urllib.parse.urlencode(params)}"
    return download_url


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and extract the Spider 1.0 dataset.")
    parser.add_argument("--gdrive-file-id", default=DEFAULT_GDRIVE_FILE_ID, help="Google Drive file ID.")
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory to extract the dataset into."
    )
    parser.add_argument(
        "--archive-path",
        type=Path,
        default=None,
        help="Path to a local zip archive. If provided, skips download entirely.",
    )
    parser.add_argument(
        "--url",
        default=None,
        help="Direct download URL. Takes precedence over --gdrive-file-id.",
    )
    args = parser.parse_args()

    output_dir: Path = args.output_dir

    if (output_dir / "train_spider.json").exists():
        print("Spider dataset already present, skipping download.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    if args.archive_path is not None:
        archive_path = args.archive_path
        cleanup_archive = False
        print(f"Using local archive: {archive_path}")
    else:
        archive_path = output_dir.parent / (output_dir.name + ".zip")
        cleanup_archive = True
        if args.url is not None:
            print(f"Downloading Spider 1.0 from {args.url} ...")
            download_url = args.url
        else:
            print(f"Resolving Google Drive download URL for file {args.gdrive_file_id} ...")
            download_url = _resolve_gdrive_url(args.gdrive_file_id)
            print("Downloading Spider 1.0 ...")
        urllib.request.urlretrieve(download_url, archive_path)

    print("Extracting...")
    with zipfile.ZipFile(archive_path) as zf:
        for member in zf.namelist():
            parts = Path(member).parts
            if len(parts) <= 1:
                continue
            target = output_dir / Path(*parts[1:])
            if member.endswith("/"):
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(zf.read(member))

    if cleanup_archive:
        archive_path.unlink()

    examples = json.loads((output_dir / "train_spider.json").read_text())
    db_count = len({ex["db_id"] for ex in examples})
    print(f"  {len(examples)} training examples across {db_count} databases.")


if __name__ == "__main__":
    main()
