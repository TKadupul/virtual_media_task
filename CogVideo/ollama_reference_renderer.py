import argparse
import ast
import json
from pathlib import Path
import re
import subprocess
import sys
import urllib.error
import urllib.request


OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
OLLAMA_MODEL = "gpt-oss:20b"

ALLOWED_IMPORTS = {
    "argparse",
    "imageio_ffmpeg",
    "math",
    "numpy",
    "pathlib",
    "PIL",
    "sys",
}

BLOCKED_CALLS = {
    "__import__",
    "breakpoint",
    "compile",
    "eval",
    "exec",
    "input",
    "open",
}


def call_ollama(instruction, num_predict=12000):
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": instruction,
        "stream": False,
        "think": False,
        "keep_alive": 0,
        "options": {
            "temperature": 0.1,
            "num_predict": num_predict,
        },
    }
    request = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=900) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Ollama HTTP {error.code}: {body}"
        ) from error
    except urllib.error.URLError as error:
        raise RuntimeError(
            f"Cannot connect to Ollama at {OLLAMA_URL}: {error}"
        ) from error
    except (TimeoutError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Invalid Ollama response: {error}") from error

    source = result.get("response", "").strip()
    if not source:
        raise RuntimeError(
            "Ollama returned no renderer source code. "
            f"done_reason={result.get('done_reason', 'unknown')}"
        )
    return strip_code_fence(source)


def strip_code_fence(source):
    source = source.strip()
    match = re.fullmatch(
        r"```(?:python)?\s*(.*?)\s*```",
        source,
        flags=re.DOTALL | re.IGNORECASE,
    )
    return match.group(1).strip() if match else source


def build_generation_instruction(user_prompt):
    return f"""
Write one complete, self-contained Python program that deterministically
renders a simple reference animation for the following video request.

USER VIDEO REQUEST:
{user_prompt}

The reference video is intended to provide exact spatial and temporal motion
to a later video-to-video model. Motion accuracy is more important than
photorealism.

PROGRAM CONTRACT:
- Return Python source code only. No Markdown or explanation.
- Use only argparse, math, pathlib, sys, numpy, Pillow, and imageio_ffmpeg.
- Do not use Blender, Manim, OpenGL, a web API, external assets, subprocess,
  the network, eval, exec, compile, open, or dynamically imported modules.
- The program must work offline and be deterministic.
- Define a main() function and an argparse interface with these exact options:
  --output_path, --num_frames, --fps, --width, and --height.
- All five options must be used. The program must create the parent directory
  of --output_path when necessary.
- Render every frame directly with Pillow and NumPy.
- Encode the frames directly to MP4 with imageio_ffmpeg.write_frames using
  codec="libx264", pix_fmt_in="rgb24", and pix_fmt_out="yuv420p".
- Do not create temporary files unless unavoidable.

VISUAL AND MOTION REQUIREMENTS:
- Construct recognizable subjects and scenery from simple shaded geometric
  primitives. Use enough distinctive parts to make front, back, left, and
  right visually distinguishable.
- Use a simple software 3D or 2.5D renderer when the request involves camera
  movement, rotation, depth, vehicles, buildings, or spatial relationships.
- Implement perspective projection and depth ordering when useful.
- Express requested motion with explicit mathematical trajectories and
  chronological keyframed phases.
- If the camera orbits, move the virtual camera around a stationary subject.
  Never fake a camera orbit by rotating the subject toward the camera.
- If a precise number of rotations is requested, calculate the angle from
  normalized frame progress so the exact total rotation is completed.
- If the camera must return to the starting viewpoint, make the first and last
  camera positions mathematically identical.
- Keep stationary objects stationary.
- If the request contains sequential actions, allocate clearly separated frame
  intervals to every action in the requested order.
- Avoid jump cuts, unexplained morphing, random motion, duplicated subjects,
  and added narrative events.
- The resulting program must be practical at 720x480, 49 frames, and 8 fps.

Before returning the code, mentally check imports, array shapes, face indices,
Pillow calls, perspective math, frame loop, and MP4 encoding.
""".strip()


def build_repair_instruction(user_prompt, source, error_message):
    return f"""
Repair the Python reference renderer below.

ORIGINAL VIDEO REQUEST:
{user_prompt}

CURRENT PROGRAM:
{source}

VALIDATION OR RUNTIME ERROR:
{error_message}

Return the complete corrected Python source only. No Markdown or explanation.
Preserve the requested scene and exact motion. Keep the same command-line
contract: --output_path, --num_frames, --fps, --width, --height.
Use only argparse, math, pathlib, sys, numpy, Pillow, and imageio_ffmpeg.
Do not use subprocess, the network, external assets, eval, exec, compile,
open, or dynamic imports.
""".strip()


def validate_source(source):
    try:
        tree = ast.parse(source)
    except SyntaxError as error:
        raise ValueError(f"Python syntax error: {error}") from error

    has_main = False
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            modules = []
            if isinstance(node, ast.Import):
                modules = [item.name.split(".")[0] for item in node.names]
            elif node.module:
                modules = [node.module.split(".")[0]]
            for module in modules:
                if module not in ALLOWED_IMPORTS:
                    raise ValueError(f"Disallowed import: {module}")

        if isinstance(node, ast.FunctionDef) and node.name == "main":
            has_main = True

        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id in BLOCKED_CALLS:
                    raise ValueError(f"Disallowed call: {node.func.id}")
            elif isinstance(node.func, ast.Attribute):
                if node.func.attr in BLOCKED_CALLS:
                    raise ValueError(f"Disallowed call: {node.func.attr}")

        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise ValueError(f"Disallowed special attribute: {node.attr}")

    if not has_main:
        raise ValueError("The generated program has no main() function.")

    required_text = [
        "--output_path",
        "--num_frames",
        "--fps",
        "--width",
        "--height",
        "imageio_ffmpeg",
        "write_frames",
    ]
    missing = [text for text in required_text if text not in source]
    if missing:
        raise ValueError(
            "Generated program is missing required elements: "
            + ", ".join(missing)
        )


def execute_renderer(
    script_path,
    output_path,
    num_frames,
    fps,
    width,
    height,
    timeout,
):
    command = [
        sys.executable,
        str(script_path),
        "--output_path",
        str(output_path),
        "--num_frames",
        str(num_frames),
        "--fps",
        str(fps),
        "--width",
        str(width),
        "--height",
        str(height),
    ]
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if result.returncode != 0:
        message = (
            f"exit_code={result.returncode}\n"
            f"stdout:\n{result.stdout[-4000:]}\n"
            f"stderr:\n{result.stderr[-8000:]}"
        )
        raise RuntimeError(message)

    if not output_path.exists() or output_path.stat().st_size == 0:
        raise RuntimeError(
            "The renderer exited successfully but created no MP4 file."
        )
    return result


def valid_name(name):
    return bool(
        name
        and Path(name).name == name
        and all(
            character in "abcdefghijklmnopqrstuvwxyz0123456789_"
            for character in name
        )
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Ask a local Ollama model to write and run a deterministic "
            "reference-video renderer."
        )
    )
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--name", default="generated_reference")
    parser.add_argument(
        "--output_dir",
        default="./tests/improve/complated",
    )
    parser.add_argument("--num_frames", type=int, default=49)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--width", type=int, default=720)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--max_repairs", type=int, default=2)
    parser.add_argument("--render_timeout", type=int, default=600)
    args = parser.parse_args()

    if not valid_name(args.name):
        parser.error(
            "--name may contain only lowercase letters, digits, and underscores."
        )
    if args.num_frames < 2:
        parser.error("--num_frames must be at least 2.")
    if args.fps < 1 or args.width < 1 or args.height < 1:
        parser.error("--fps, --width, and --height must be positive.")
    if args.max_repairs < 0:
        parser.error("--max_repairs cannot be negative.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    script_path = output_dir / f"{args.name}_generated_renderer.py"
    output_path = output_dir / f"{args.name}_original.mp4"
    record_path = output_dir / f"{args.name}_renderer_record.json"

    try:
        print("Requesting renderer source from Ollama...", file=sys.stderr)
        source = call_ollama(build_generation_instruction(args.prompt))

        attempt_records = []
        for attempt in range(args.max_repairs + 1):
            script_path.write_text(source + "\n", encoding="utf-8")
            try:
                validate_source(source)
                print(
                    f"Running generated renderer (attempt {attempt + 1})...",
                    file=sys.stderr,
                )
                result = execute_renderer(
                    script_path=script_path,
                    output_path=output_path,
                    num_frames=args.num_frames,
                    fps=args.fps,
                    width=args.width,
                    height=args.height,
                    timeout=args.render_timeout,
                )
                attempt_records.append(
                    {
                        "attempt": attempt + 1,
                        "status": "success",
                        "stdout": result.stdout[-4000:],
                        "stderr": result.stderr[-4000:],
                    }
                )
                break
            except (ValueError, RuntimeError, subprocess.TimeoutExpired) as error:
                error_text = str(error)
                attempt_records.append(
                    {
                        "attempt": attempt + 1,
                        "status": "failed",
                        "error": error_text,
                    }
                )
                if attempt >= args.max_repairs:
                    raise RuntimeError(
                        "The generated renderer still failed after "
                        f"{attempt + 1} attempt(s): {error_text}"
                    ) from error

                print(
                    f"Renderer attempt {attempt + 1} failed; "
                    "requesting an automatic repair...",
                    file=sys.stderr,
                )
                source = call_ollama(
                    build_repair_instruction(
                        user_prompt=args.prompt,
                        source=source,
                        error_message=error_text,
                    )
                )

        record = {
            "prompt": args.prompt,
            "model": OLLAMA_MODEL,
            "generated_script": str(script_path),
            "output_video": str(output_path),
            "num_frames": args.num_frames,
            "fps": args.fps,
            "width": args.width,
            "height": args.height,
            "attempts": attempt_records,
        }
        record_path.write_text(
            json.dumps(record, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    except RuntimeError as error:
        print(error, file=sys.stderr)
        sys.exit(1)

    print(output_path)
    print(script_path)
    print(record_path)


if __name__ == "__main__":
    main()
