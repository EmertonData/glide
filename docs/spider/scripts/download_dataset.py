import json
import re
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

GDRIVE_FILE_ID = "1403EGqzIDoHMdQF4c9Bkyl7dZLZ5Wt6J"
OUTPUT_DIR = Path("data/spider")
ARCHIVE_PATH = Path("data/spider.zip")


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
    if (OUTPUT_DIR / "train_spider.json").exists():
        print("Spider dataset already present, skipping download.")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Resolving Google Drive download URL for file {GDRIVE_FILE_ID} ...")
    download_url = _resolve_gdrive_url(GDRIVE_FILE_ID)

    print("Downloading Spider 1.0 ...")
    urllib.request.urlretrieve(download_url, ARCHIVE_PATH)

    print("Extracting...")
    with zipfile.ZipFile(ARCHIVE_PATH) as zf:
        for member in zf.namelist():
            parts = Path(member).parts
            if len(parts) <= 1:
                continue
            target = OUTPUT_DIR / Path(*parts[1:])
            if member.endswith("/"):
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(zf.read(member))

    ARCHIVE_PATH.unlink()

    examples = json.loads((OUTPUT_DIR / "train_spider.json").read_text())
    db_count = len({ex["db_id"] for ex in examples})
    print(f"  {len(examples)} training examples across {db_count} databases.")


if __name__ == "__main__":
    main()
