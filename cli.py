#!/usr/bin/env python3
"""Run the bundled ComfyUI img2img graph through its local HTTP API.

Python 3.9+, standard library only. No checkpoints are distributed.
"""

from __future__ import annotations

import argparse
import copy
import json
import mimetypes
import secrets
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parent
DEFAULT_API = ROOT / "api" / "prompt.json"
EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


class ComfyError(RuntimeError):
    pass


def request(server: str, path: str, *, payload: bytes | None = None,
            content_type: str | None = None, timeout: float = 30) -> bytes:
    url = server.rstrip("/") + path
    headers = {"Accept": "application/json"} if payload is not None else {}
    if content_type:
        headers["Content-Type"] = content_type
    req = urllib.request.Request(url, data=payload, headers=headers,
                                 method="POST" if payload is not None else "GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return response.read()
    except urllib.error.HTTPError as exc:
        details = exc.read().decode("utf-8", "replace")[:2000]
        raise ComfyError(f"HTTP {exc.code} at {url}: {details}") from exc
    except urllib.error.URLError as exc:
        raise ComfyError(f"Cannot connect to {url}: {exc.reason}") from exc


def request_json(server: str, path: str, data: dict | None = None) -> dict:
    content = json.dumps(data).encode("utf-8") if data is not None else None
    raw = request(server, path, payload=content,
                  content_type="application/json" if data is not None else None)
    return json.loads(raw)


def upload_image(server: str, file: Path) -> str:
    # Use a random filename so concurrent jobs cannot overwrite another upload.
    name = f"img2img-{uuid4().hex}{file.suffix.lower()}"
    boundary = f"----comfy-{uuid4().hex}"
    mime = mimetypes.guess_type(name)[0] or "application/octet-stream"
    header = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="image"; filename="{name}"\r\n'
        f"Content-Type: {mime}\r\n\r\n"
    ).encode("ascii")
    payload = header + file.read_bytes() + f"\r\n--{boundary}--\r\n".encode("ascii")
    data = request(server, "/upload/image", payload=payload,
                   content_type=f"multipart/form-data; boundary={boundary}")
    uploaded = json.loads(data)
    if not uploaded.get("name"):
        raise ComfyError(f"Unexpected upload response: {uploaded}")
    if uploaded.get("subfolder"):
        return f"{uploaded['subfolder']}/{uploaded['name']}"
    return uploaded["name"]


def get_images_from_history(history: dict, prompt_id: str) -> list[dict] | None:
    entry = history.get(prompt_id)
    if entry is None:
        return None
    images = []
    for value in entry.get("outputs", {}).values():
        images.extend(value.get("images", []))
    if images:
        return images
    status = entry.get("status", {})
    if status.get("status_str") == "error" or status.get("completed") is False:
        raise ComfyError(f"ComfyUI execution failed: {json.dumps(status, ensure_ascii=False)[:2000]}")
    raise ComfyError(f"Job {prompt_id} completed without images: {json.dumps(status, ensure_ascii=False)}")


def wait_for_images(server: str, prompt_id: str, timeout: float, poll: float) -> list[dict]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        history = request_json(server, f"/history/{urllib.parse.quote(prompt_id)}")
        images = get_images_from_history(history, prompt_id)
        if images is not None:
            return images
        time.sleep(poll)
    raise ComfyError(f"Timeout after {timeout:g}s waiting for job {prompt_id}")


def build_prompt(template: dict, args: argparse.Namespace, image_name: str) -> dict:
    prompt = copy.deepcopy(template)
    prompt["1"]["inputs"]["image"] = image_name
    prompt["9"]["inputs"]["unet_name"] = args.model or prompt["9"]["inputs"]["unet_name"]
    prompt["12"]["inputs"]["clip_name"] = args.clip or prompt["12"]["inputs"]["clip_name"]
    prompt["2"]["inputs"]["vae_name"] = args.vae or prompt["2"]["inputs"]["vae_name"]
    prompt["10"]["inputs"]["text"] = args.prompt
    prompt["13"]["inputs"].update(
        steps=args.steps,
        denoise=args.denoise,
        seed=args.seed if args.seed is not None else secrets.randbits(53),
        scheduler=args.scheduler,
    )
    prompt["14"]["inputs"]["shift"] = args.shift
    return prompt


def collect_inputs(path: Path) -> list[Path]:
    if path.is_file():
        if path.suffix.lower() not in EXTENSIONS:
            raise ComfyError(f"Unsupported image extension: {path.suffix}")
        return [path]
    if path.is_dir():
        images = sorted(p for p in path.iterdir()
                        if p.is_file() and p.suffix.lower() in EXTENSIONS)
        if not images:
            raise ComfyError(f"No images in {path}")
        return images
    raise ComfyError(f"Input not found: {path}")


def target_for(source: Path, output: Path | None, is_batch: bool) -> Path:
    if is_batch:
        directory = output or Path("output")
        return directory / f"{source.stem}_processed.png"
    if output is None:
        return source.with_name(f"{source.stem}_processed.png")
    if output.suffix.lower() == ".png":
        return output
    return output / f"{source.stem}_processed.png"


def run(args: argparse.Namespace) -> int:
    if args.steps < 1:
        raise ComfyError("--steps must be >= 1")
    if not 0 < args.denoise <= 1:
        raise ComfyError("--denoise must be in (0, 1]")
    if args.seed is not None and not (0 <= args.seed < (1 << 64)):
        raise ComfyError("--seed must be a non-negative 64-bit integer")
    if args.timeout <= 0 or args.poll <= 0:
        raise ComfyError("--timeout and --poll must be positive")
    if not args.server.startswith(("http://", "https://")):
        raise ComfyError("--server must start with http:// or https://")

    sources = collect_inputs(args.input)
    is_batch = args.input.is_dir()
    if is_batch and args.output and args.output.suffix.lower() == ".png":
        raise ComfyError("--output must be a directory when --input is a directory")
    template = json.loads(args.api_prompt.read_text(encoding="utf-8"))
    for i, source in enumerate(sources, start=1):
        target = target_for(source, args.output, is_batch)
        if target.resolve() == source.resolve():
            raise ComfyError("Refusing to overwrite the input image")
        print(f"[{i}/{len(sources)}] {source.name}", flush=True)
        image_name = upload_image(args.server, source)
        prompt = build_prompt(template, args, image_name)
        submitted = request_json(args.server, "/prompt", {"prompt": prompt, "client_id": str(uuid4())})
        prompt_id = submitted.get("prompt_id")
        if not prompt_id:
            raise ComfyError(f"ComfyUI refused prompt: {json.dumps(submitted, ensure_ascii=False)[:2000]}")
        images = wait_for_images(args.server, prompt_id, args.timeout, args.poll)
        image = images[0]
        query = urllib.parse.urlencode({
            "filename": image["filename"],
            "subfolder": image.get("subfolder", ""),
            "type": image.get("type", "output"),
        })
        result = request(args.server, "/view?" + query)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(result)
        print(f"  saved: {target} ({len(result)} bytes)", flush=True)
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch-capable ComfyUI img2img runner for the included workflow."
    )
    parser.add_argument("--input", "-i", type=Path, required=True, help="Image file or flat directory")
    parser.add_argument("--output", "-o", type=Path, help="PNG file (single input) or directory")
    parser.add_argument("--server", default="http://127.0.0.1:8188")
    parser.add_argument("--api-prompt", type=Path, default=DEFAULT_API)
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--denoise", type=float, default=0.6)
    parser.add_argument("--seed", type=int, help="Fixed seed; random for each image if omitted")
    parser.add_argument("--scheduler", default="karras", help="e.g. karras or simple")
    parser.add_argument("--shift", type=float, default=3.0)
    parser.add_argument("--prompt", default="", help="Positive prompt; defaults to empty")
    parser.add_argument("--model", help="Override diffusion checkpoint filename")
    parser.add_argument("--clip", help="Override Qwen text encoder filename")
    parser.add_argument("--vae", help="Override VAE filename")
    parser.add_argument("--timeout", type=float, default=600, help="Per-image timeout in seconds")
    parser.add_argument("--poll", type=float, default=1.0, help="Polling interval in seconds")
    return parser.parse_args(argv)


if __name__ == "__main__":
    try:
        sys.exit(run(parse_args()))
    except (ComfyError, OSError, ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
