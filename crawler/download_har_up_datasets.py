import argparse
import html
import json
import re
from pathlib import Path
from urllib.parse import urljoin

import requests


DEFAULT_LINKS_FILE = Path(__file__).with_name("har_up_dataset_links.json")
DEFAULT_OUTPUT_DIR = Path.cwd() / "DataSetCSV"
CHUNK_SIZE = 1024 * 1024


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download HAR-UP DataSet CSV files from har_up_dataset_links.json."
    )
    parser.add_argument(
        "--links-file",
        type=Path,
        default=DEFAULT_LINKS_FILE,
        help=f"JSON file containing DataSet links. Default: {DEFAULT_LINKS_FILE}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory to store CSV files. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Download again even when the target CSV already exists.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="Request timeout in seconds. Default: 60",
    )
    return parser.parse_args()


def load_links(path):
    with path.open("r", encoding="utf-8") as f:
        links = json.load(f)

    if not isinstance(links, list):
        raise ValueError(f"{path} must contain a JSON list.")

    return links


def target_filename(item):
    subject = item["subject"]
    activity = item["activity"]
    trial = item["trial"]
    return f"{subject}{activity}{trial}.csv"


def google_drive_confirm_url(response):
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            return response.url + ("&" if "?" in response.url else "?") + f"confirm={value}"

    content_type = response.headers.get("content-type", "")
    if "text/html" not in content_type.lower():
        return None

    page = response.text
    match = re.search(r'href="([^"]*?/uc\?[^"]*?confirm=[^"]+)"', page)
    if not match:
        return None

    confirm_href = html.unescape(match.group(1))
    return urljoin(response.url, confirm_href)


def get_download_response(session, url, timeout):
    response = session.get(url, stream=True, timeout=timeout)
    response.raise_for_status()

    confirm_url = google_drive_confirm_url(response)
    if not confirm_url:
        return response

    response.close()
    confirmed = session.get(confirm_url, stream=True, timeout=timeout)
    confirmed.raise_for_status()
    return confirmed


def download_file(session, url, target_path, timeout):
    target_path.parent.mkdir(parents=True, exist_ok=True)
    part_path = target_path.with_suffix(target_path.suffix + ".part")

    with get_download_response(session, url, timeout) as response:
        content_type = response.headers.get("content-type", "")
        with part_path.open("wb") as f:
            for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                if chunk:
                    f.write(chunk)

    if part_path.stat().st_size == 0:
        part_path.unlink(missing_ok=True)
        raise RuntimeError(f"Downloaded an empty file: {target_path}")

    part_path.replace(target_path)

    if "text/html" in content_type.lower():
        print(f"[WARN] {target_path.name} may be an HTML page, not a CSV.")


def main():
    args = parse_args()
    links = load_links(args.links_file)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    downloaded = 0
    skipped = 0
    failed = 0

    for index, item in enumerate(links, start=1):
        url = item.get("url")
        if not url:
            failed += 1
            print(f"[{index}/{len(links)}] Missing URL: {item}")
            continue

        target_path = args.output_dir / target_filename(item)
        if target_path.exists() and not args.overwrite:
            skipped += 1
            print(f"[{index}/{len(links)}] Skip existing: {target_path}")
            continue

        print(f"[{index}/{len(links)}] Downloading: {target_path}")
        try:
            download_file(session, url, target_path, args.timeout)
            downloaded += 1
        except Exception as exc:
            failed += 1
            print(f"    [ERROR] {type(exc).__name__}: {exc}")

    print("=" * 60)
    print(f"Done. Downloaded: {downloaded}, skipped: {skipped}, failed: {failed}")
    print(f"Output directory: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
