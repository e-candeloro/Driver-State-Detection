import argparse
import hashlib
import os
import tempfile
import urllib.request
from pathlib import Path

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
MODEL_SHA256 = "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff"
MAX_MODEL_BYTES = 10 * 1024 * 1024
DEFAULT_MODEL_PATH = Path("models/face_landmarker.task")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as model_file:
        for chunk in iter(lambda: model_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_model(destination: Path = DEFAULT_MODEL_PATH, force: bool = False) -> Path:
    """Atomically download and checksum-verify the pinned Face Landmarker model."""
    destination = Path(destination)
    if destination.exists() and not force:
        if sha256(destination) == MODEL_SHA256:
            return destination
        raise RuntimeError(
            f"{destination} exists but has the wrong checksum; use --force to replace it"
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent, prefix=f".{destination.name}.", delete=False
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            with urllib.request.urlopen(MODEL_URL, timeout=120) as response:
                content_length = response.headers.get("Content-Length")
                if content_length and int(content_length) > MAX_MODEL_BYTES:
                    raise RuntimeError("Model download is larger than the allowed size")
                downloaded_bytes = 0
                while chunk := response.read(1024 * 1024):
                    downloaded_bytes += len(chunk)
                    if downloaded_bytes > MAX_MODEL_BYTES:
                        raise RuntimeError(
                            "Model download is larger than the allowed size"
                        )
                    temporary_file.write(chunk)

        actual_checksum = sha256(temporary_path)
        if actual_checksum != MODEL_SHA256:
            raise RuntimeError(
                "Downloaded model checksum mismatch: "
                f"expected {MODEL_SHA256}, got {actual_checksum}"
            )
        os.replace(temporary_path, destination)
        return destination
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def main():
    parser = argparse.ArgumentParser(description="Download the Face Landmarker model")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help=f"Model destination (default: {DEFAULT_MODEL_PATH})",
    )
    parser.add_argument(
        "--force", action="store_true", help="Replace an existing model"
    )
    args = parser.parse_args()

    try:
        model_path = download_model(args.output, args.force)
    except (OSError, RuntimeError) as error:
        parser.exit(1, f"Model download failed: {error}\n")
    print(f"Model ready: {model_path}")


if __name__ == "__main__":
    main()
