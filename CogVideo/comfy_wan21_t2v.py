import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request


DEFAULT_COMFY_URL = "http://127.0.0.1:8188"

DEFAULT_NEGATIVE_PROMPT = (
    "oversaturated, overexposed, static image, blurry details, subtitles, "
    "painting, low quality, jpeg artifacts, ugly, deformed, disfigured, "
    "malformed limbs, fused fingers, duplicate subjects, extra objects, "
    "crowded background, camera cuts, fisheye distortion"
)


def api_request(url, method="GET", payload=None, timeout=300):
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    request = urllib.request.Request(
        url,
        data=data,
        headers=headers,
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            content_type = response.headers.get("Content-Type", "")
            body = response.read()
            if "application/json" in content_type:
                return json.loads(body.decode("utf-8"))
            return body
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"ComfyUI HTTP {error.code}: {body}"
        ) from error
    except urllib.error.URLError as error:
        raise RuntimeError(
            f"Cannot connect to ComfyUI at {url}: {error}"
        ) from error


def build_workflow(
    prompt,
    negative_prompt,
    seed,
    frames,
    steps,
    model_variant,
):
    if model_variant == "wan2.2":
        unet_name = "wan2.2_ti2v_5B_fp16.safetensors"
        vae_name = "wan2.2_vae.safetensors"
        width = 1280
        height = 704
        cfg = 5.0
        fps = 24.0
        latent_node = {
            "class_type": "Wan22ImageToVideoLatent",
            "inputs": {
                "vae": ["39", 0],
                "width": width,
                "height": height,
                "length": frames,
                "batch_size": 1,
            },
        }
        filename_prefix = "api/wan22_t2v"
    else:
        unet_name = "wan2.1_t2v_1.3B_fp16.safetensors"
        vae_name = "wan_2.1_vae.safetensors"
        width = 832
        height = 480
        cfg = 6.0
        fps = 16.0
        latent_node = {
            "class_type": "EmptyHunyuanLatentVideo",
            "inputs": {
                "width": width,
                "height": height,
                "length": frames,
                "batch_size": 1,
            },
        }
        filename_prefix = "api/wan21_t2v"

    return {
        "37": {
            "class_type": "UNETLoader",
            "inputs": {
                "unet_name": unet_name,
                "weight_dtype": "default",
            },
        },
        "38": {
            "class_type": "CLIPLoader",
            "inputs": {
                "clip_name": "umt5_xxl_fp8_e4m3fn_scaled.safetensors",
                "type": "wan",
                "device": "default",
            },
        },
        "39": {
            "class_type": "VAELoader",
            "inputs": {
                "vae_name": vae_name,
            },
        },
        "48": {
            "class_type": "ModelSamplingSD3",
            "inputs": {
                "model": ["37", 0],
                "shift": 8.0,
            },
        },
        "6": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": prompt,
                "clip": ["38", 0],
            },
        },
        "7": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": negative_prompt,
                "clip": ["38", 0],
            },
        },
        "40": latent_node,
        "3": {
            "class_type": "KSampler",
            "inputs": {
                "seed": seed,
                "steps": steps,
                "cfg": cfg,
                "sampler_name": "uni_pc",
                "scheduler": "simple",
                "denoise": 1.0,
                "model": ["48", 0],
                "positive": ["6", 0],
                "negative": ["7", 0],
                "latent_image": ["40", 0],
            },
        },
        "8": {
            "class_type": "VAEDecode",
            "inputs": {
                "samples": ["3", 0],
                "vae": ["39", 0],
            },
        },
        "47": {
            "class_type": "SaveWEBM",
            "inputs": {
                "images": ["8", 0],
                "filename_prefix": filename_prefix,
                "codec": "vp9",
                "fps": fps,
                "crf": 32,
            },
        },
    }


def check_nodes(base_url):
    object_info = api_request(f"{base_url}/object_info")
    common_nodes = {
        "UNETLoader",
        "CLIPLoader",
        "VAELoader",
        "ModelSamplingSD3",
        "CLIPTextEncode",
        "KSampler",
        "VAEDecode",
        "SaveWEBM",
    }
    missing = sorted(common_nodes.difference(object_info))
    if missing:
        raise RuntimeError(
            "ComfyUI is missing required nodes: " + ", ".join(missing)
        )

    unet_names = (
        object_info.get("UNETLoader", {})
        .get("input", {})
        .get("required", {})
        .get("unet_name", [[]])[0]
    )
    vae_names = (
        object_info.get("VAELoader", {})
        .get("input", {})
        .get("required", {})
        .get("vae_name", [[]])[0]
    )

    if (
        "wan2.2_ti2v_5B_fp16.safetensors" in unet_names
        and "wan2.2_vae.safetensors" in vae_names
    ):
        if "Wan22ImageToVideoLatent" not in object_info:
            raise RuntimeError(
                "Wan2.2 is installed, but Wan22ImageToVideoLatent is missing."
            )
        return "wan2.2"

    if (
        "wan2.1_t2v_1.3B_fp16.safetensors" in unet_names
        and "wan_2.1_vae.safetensors" in vae_names
    ):
        if "EmptyHunyuanLatentVideo" not in object_info:
            raise RuntimeError(
                "Wan2.1 is installed, but EmptyHunyuanLatentVideo is missing."
            )
        return "wan2.1"

    raise RuntimeError(
        "No supported local Wan model was found. Available diffusion models: "
        + ", ".join(unet_names)
    )


def queue_workflow(base_url, workflow):
    response = api_request(
        f"{base_url}/prompt",
        method="POST",
        payload={"prompt": workflow},
    )
    prompt_id = response.get("prompt_id")
    if not prompt_id:
        raise RuntimeError(f"ComfyUI returned no prompt_id: {response}")
    return prompt_id


def wait_for_history(base_url, prompt_id, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        history = api_request(f"{base_url}/history/{prompt_id}")
        item = history.get(prompt_id)
        if item:
            status = item.get("status", {})
            if status.get("status_str") == "error":
                raise RuntimeError(
                    "ComfyUI generation failed: "
                    + json.dumps(status, ensure_ascii=False)
                )
            if status.get("completed") or item.get("outputs"):
                return item
        time.sleep(5)
    raise RuntimeError(
        f"ComfyUI generation exceeded the {timeout}-second timeout."
    )


def find_output_file(history_item):
    candidates = []

    def visit(value):
        if isinstance(value, dict):
            if "filename" in value:
                candidates.append(value)
            for nested in value.values():
                visit(nested)
        elif isinstance(value, list):
            for nested in value:
                visit(nested)

    visit(history_item.get("outputs", {}))
    if not candidates:
        raise RuntimeError("ComfyUI history contains no generated file.")

    for candidate in candidates:
        filename = str(candidate.get("filename", "")).lower()
        if filename.endswith((".webm", ".mp4", ".webp", ".gif")):
            return candidate
    return candidates[0]


def download_output(base_url, output_info, destination):
    query = urllib.parse.urlencode(
        {
            "filename": output_info["filename"],
            "subfolder": output_info.get("subfolder", ""),
            "type": output_info.get("type", "output"),
        }
    )
    content = api_request(f"{base_url}/view?{query}", timeout=600)
    destination.write_bytes(content)
    if destination.stat().st_size == 0:
        raise RuntimeError("Downloaded ComfyUI output is empty.")


def convert_to_mp4(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        encoder_result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as error:
        raise RuntimeError("ffmpeg is not installed.") from error
    except subprocess.CalledProcessError as error:
        raise RuntimeError("Unable to inspect FFmpeg encoders.") from error

    available_encoders = encoder_result.stdout
    encoder_options = None
    for encoder_name, extra_options in (
        ("libx264", []),
        ("libopenh264", ["-b:v", "8M"]),
        ("h264_nvenc", ["-preset", "medium", "-cq", "22"]),
        ("mpeg4", ["-q:v", "3"]),
    ):
        if encoder_name in available_encoders:
            encoder_options = ["-c:v", encoder_name, *extra_options]
            print(
                f"Using FFmpeg encoder: {encoder_name}",
                file=sys.stderr,
            )
            break

    if encoder_options is None:
        raise RuntimeError(
            "FFmpeg has no supported MP4 video encoder."
        )

    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(source),
        *encoder_options,
        "-pix_fmt",
        "yuv420p",
        "-an",
        str(destination),
    ]
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as error:
        raise RuntimeError(
            f"ffmpeg conversion failed with exit code {error.returncode}."
        ) from error


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate a local Wan2.1 or Wan2.2 T2V video through ComfyUI."
        )
    )
    parser.add_argument("--prompt", required=True)
    parser.add_argument(
        "--output_path",
        default="./tests/improve/complated/wan_original.mp4",
    )
    parser.add_argument(
        "--negative_prompt",
        default=DEFAULT_NEGATIVE_PROMPT,
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--frames",
        type=int,
        default=None,
        help=(
            "Must use 4n+1. Defaults to 121 for Wan2.2 and 81 for Wan2.1."
        ),
    )
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--comfy_url", default=DEFAULT_COMFY_URL)
    args = parser.parse_args()

    if (
        args.frames is not None
        and (args.frames < 1 or (args.frames - 1) % 4 != 0)
    ):
        parser.error("--frames must use the form 4n+1, for example 33 or 81.")
    if args.steps is not None and args.steps < 1:
        parser.error("--steps must be positive.")

    base_url = args.comfy_url.rstrip("/")
    output_path = Path(args.output_path)
    temporary_path = output_path.with_suffix(".download.webm")

    try:
        model_variant = check_nodes(base_url)
        frames = args.frames
        if frames is None:
            frames = 121 if model_variant == "wan2.2" else 81
        steps = args.steps
        if steps is None:
            steps = 20 if model_variant == "wan2.2" else 30
        print(
            f"Detected {model_variant}: frames={frames}, steps={steps}",
            file=sys.stderr,
        )
        workflow = build_workflow(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            seed=args.seed,
            frames=frames,
            steps=steps,
            model_variant=model_variant,
        )
        prompt_id = queue_workflow(base_url, workflow)
        print(f"Queued ComfyUI prompt: {prompt_id}", file=sys.stderr)

        history_item = wait_for_history(
            base_url,
            prompt_id,
            args.timeout,
        )
        output_info = find_output_file(history_item)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        download_output(base_url, output_info, temporary_path)
        convert_to_mp4(temporary_path, output_path)
        temporary_path.unlink(missing_ok=True)
    except RuntimeError as error:
        print(error, file=sys.stderr)
        sys.exit(1)

    print(output_path)


if __name__ == "__main__":
    main()
