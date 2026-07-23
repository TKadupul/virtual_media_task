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
MODEL = "gpt-oss:20b"

# These profiles describe visible appearance rather than claiming a person's
# ethnicity from appearance. They span skin tone, hair texture and colour,
# facial hair, and adult age.
APPEARANCE_PROFILES = [
    "deep dark-brown skin, a shaved head, and a clean-shaven face",
    "dark-brown skin, short tightly curled black hair, and a light moustache",
    "deep-brown skin, short black locs, and a neatly trimmed goatee",
    "dark skin, close-cropped coiled hair, and a short full beard",
    "warm brown skin, short black twists, and a clean-shaven face",
    "medium-brown skin, thick wavy black hair, and a neat moustache",
    "golden-brown skin, short curly black hair, and light stubble",
    "tan skin, straight shoulder-length black hair, and a clean-shaven face",
    "warm tan skin, short straight black hair, and a sparse moustache",
    "light-brown skin, medium-length wavy dark hair, and a short beard",
    "olive skin, short wavy black hair, and a neatly trimmed beard",
    "light olive skin, thick curly dark-brown hair, and a clean-shaven face",
    "medium olive skin, a shaved head, and a dark full beard",
    "fair skin, short straight black hair, and a clean-shaven face",
    "fair skin, short dark-brown hair, and a light moustache",
    "light skin, shoulder-length wavy brown hair, and a clean-shaven face",
    "fair skin with freckles, short red hair, and a trimmed red beard",
    "pale skin, short blond hair, and a clean-shaven face",
    "light skin, sandy-blond wavy hair, and light stubble",
    "medium skin, short salt-and-pepper hair, and a matching short beard",
    "deep-brown skin, short gray hair, and a clean-shaven face",
    "dark skin, a bald head, and a full gray beard",
    "warm brown skin, long curly black hair tied back, and a clean-shaven face",
    "tan skin, long straight dark hair tied back, and a light moustache",
    "olive skin, short silver hair, and a neatly trimmed gray moustache",
    "light olive skin, a bald head, and a clean-shaven face",
    "medium-brown skin, short auburn curls, and a short beard",
    "golden-tan skin, thick shoulder-length wavy black hair, and light stubble",
    "fair skin, short silver hair, and a clean-shaven mature face",
    "deep dark-brown skin, medium-length natural curls, and a short moustache",
]

PROFILE_MULTIPLIER = 17
PROFILE_OFFSET = 11


def select_profile(seed):
    """Select one profile deterministically from the CogVideoX seed."""
    index = (
        seed * PROFILE_MULTIPLIER + PROFILE_OFFSET
    ) % len(APPEARANCE_PROFILES)
    return index, APPEARANCE_PROFILES[index]


def build_instruction(mode, prompt, seed=None):
    if mode == "bias":
        profile_index, required_appearance = select_profile(seed)

        print(
            "Bias profile selection: "
            f"seed={seed}, "
            f"profile={profile_index + 1}/{len(APPEARANCE_PROFILES)}, "
            "algorithm=(seed * 17 + 11) % 30",
            file=sys.stderr,
        )
        print(
            f"Required appearance: {required_appearance}",
            file=sys.stderr,
        )

        instruction = f"""
Rewrite the following input as a concise English video-generation prompt
for CogVideoX-5B.

Original input:
{prompt}

Required visible appearance:
{required_appearance}

Requirements:
- Write between 35 and 60 English words.
- Preserve the original number of subjects, action, setting, and intention.
- Clearly incorporate every required visible appearance attribute.
- Describe the person respectfully and without stereotypes.
- Use a fixed camera and simple, stable lighting.
- Do not add people, objects, actions, locations, or narrative events.
- Avoid unnecessary cinematic language and composition terminology.
- Do not use contradictory descriptions.
- End with exactly: No other people or objects appear.
- Return only the final English prompt.
- Do not include analysis, labels, quotation marks, or Markdown.
"""
        return instruction.strip()

    if mode == "complated":
        instruction = f"""
Rewrite the following input as a concise English video-to-video prompt
for CogVideoX-5B.

Original input:
{prompt}

The reference video already provides the requested camera trajectory and
temporal motion.

Requirements:
- Write between 35 and 70 English words.
- Preserve the original subject, setting, camera movement, and intention.
- Explicitly tell the model to preserve the reference video's motion.
- Keep the central subject stationary and visually consistent when requested.
- Use simple, concrete, and visually observable descriptions.
- Do not add people, objects, actions, locations, cuts, or narrative events.
- Avoid unnecessary cinematic language and composition terminology.
- Do not use contradictory descriptions.
- Return only the final English prompt.
- Do not include analysis, labels, quotation marks, or Markdown.
"""
        return instruction.strip()

    instruction = f"""
Rewrite the following input as a concise English video-generation prompt
for CogVideoX-5B.

Original input:
{prompt}

Requirements:
- Write between 35 and 60 English words.
- Preserve the original number of subjects, action, setting, and intention.
- If several actions are present, describe them in chronological order.
- Use simple, concrete, and visually observable descriptions.
- Use a stable camera unless camera movement is explicitly requested.
- Do not add people, objects, actions, locations, or narrative events.
- Avoid unnecessary cinematic language and composition terminology.
- Do not use contradictory descriptions.
- Return only the final English prompt.
- Do not include analysis, labels, quotation marks, or Markdown.
"""
    return instruction.strip()


def call_ollama(instruction, temperature=0.2, num_predict=512):
    request_data = {
        "model": MODEL,
        "prompt": instruction,
        "stream": False,
        "think": "low",
        "keep_alive": 0,
        "options": {
            "temperature": temperature,
            "num_predict": num_predict,
        },
    }

    request = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(request_data).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            response_text = response.read().decode("utf-8")
            return json.loads(response_text)

    except urllib.error.HTTPError as error:
        error_body = error.read().decode("utf-8", errors="replace")
        print(
            f"Ollama HTTP error {error.code}: {error_body}",
            file=sys.stderr,
        )
        sys.exit(1)

    except urllib.error.URLError as error:
        print(
            f"Unable to connect to Ollama at {OLLAMA_URL}: {error}",
            file=sys.stderr,
        )
        sys.exit(1)

    except TimeoutError:
        print("The Ollama request timed out.", file=sys.stderr)
        sys.exit(1)

    except json.JSONDecodeError:
        print("Ollama returned invalid JSON.", file=sys.stderr)
        sys.exit(1)


def optimize_prompt(prompt, mode, seed=None):
    instruction = build_instruction(
        mode=mode,
        prompt=prompt,
        seed=seed,
    )
    result = call_ollama(instruction)

    optimized_prompt = result.get("response", "").strip()
    optimized_prompt = optimized_prompt.strip('"').strip()

    if not optimized_prompt:
        done_reason = result.get("done_reason", "unknown")
        thinking_length = len(result.get("thinking", ""))
        print(
            "Ollama returned no final answer. "
            f"done_reason={done_reason}, "
            f"thinking_characters={thinking_length}",
            file=sys.stderr,
        )
        sys.exit(1)

    word_count = len(optimized_prompt.split())
    print(f"Optimized prompt length: {word_count} words", file=sys.stderr)

    maximum_words = 70 if mode == "complated" else 60
    if word_count < 35 or word_count > maximum_words:
        print(
            "Warning: the optimized prompt is outside the requested "
            f"35-{maximum_words} word range.",
            file=sys.stderr,
        )

    return optimized_prompt


def run_command(command, description, cwd=None, timeout=None):
    print(f"\n{description}", file=sys.stderr)
    print(" ".join(str(part) for part in command), file=sys.stderr)
    try:
        subprocess.run(
            command,
            check=True,
            cwd=cwd,
            timeout=timeout,
        )
    except FileNotFoundError as error:
        print(f"Unable to start command: {error}", file=sys.stderr)
        sys.exit(1)
    except subprocess.CalledProcessError as error:
        print(
            f"{description} failed with exit code {error.returncode}.",
            file=sys.stderr,
        )
        sys.exit(error.returncode)
    except subprocess.TimeoutExpired:
        print(
            f"{description} exceeded the {timeout}-second time limit.",
            file=sys.stderr,
        )
        sys.exit(1)


def build_renderer_instruction(prompt, num_frames, fps, width, height):
    return f"""
Create a complete, self-contained Python program that procedurally renders
a simple reference animation matching the user's request.

User request:
{prompt}

Technical requirements:
- The animation is a motion reference for CogVideoX video-to-video generation.
- Prioritize correct object motion, camera motion, action order, and timing
  over visual realism.
- Generate exactly {num_frames} frames at {fps} fps.
- Every frame must be exactly {width} by {height} pixels.
- The first command-line argument, sys.argv[1], is the required MP4 output path.
- Write a playable H.264 MP4 to that exact path.
- Use only Python's standard library plus numpy, Pillow, imageio, or
  imageio_ffmpeg.
- Use deterministic mathematics; do not use uncontrolled randomness.
- Do not access the network.
- Do not read input files.
- Do not invoke shell commands or subprocesses.
- Do not access files other than the requested output video.
- Do not use Blender, OpenCV, moviepy, matplotlib, or external assets.
- Keep the program reasonably short and able to finish within two minutes.
- Include all rendering and MP4 encoding logic in the program.
- Return only executable Python source code.
- Do not use Markdown fences or provide explanations.
"""


def extract_python_source(response):
    source = response.strip()
    fenced = re.fullmatch(
        r"```(?:python)?\s*(.*?)\s*```",
        source,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if fenced:
        source = fenced.group(1).strip()
    return source


def validate_generated_renderer(source):
    try:
        tree = ast.parse(source)
    except SyntaxError as error:
        raise ValueError(f"Generated renderer has invalid syntax: {error}")

    allowed_import_roots = {
        "math",
        "sys",
        "typing",
        "numpy",
        "PIL",
        "imageio",
        "imageio_ffmpeg",
    }
    forbidden_calls = {
        "__import__",
        "breakpoint",
        "compile",
        "eval",
        "exec",
        "input",
        "open",
    }
    forbidden_names = {
        "os",
        "pathlib",
        "requests",
        "shutil",
        "socket",
        "subprocess",
        "urllib",
    }

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".", 1)[0]
                if root not in allowed_import_roots:
                    raise ValueError(
                        f"Generated renderer imports forbidden module: "
                        f"{alias.name}"
                    )

        if isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".", 1)[0]
            if root not in allowed_import_roots:
                raise ValueError(
                    f"Generated renderer imports forbidden module: "
                    f"{node.module}"
                )

        if isinstance(node, ast.Name):
            if node.id in forbidden_names:
                raise ValueError(
                    f"Generated renderer uses forbidden name: {node.id}"
                )

        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in forbidden_calls:
                raise ValueError(
                    f"Generated renderer uses forbidden function: "
                    f"{node.func.id}"
                )

    forbidden_text = (
        "http://",
        "https://",
        "../",
        "/etc/",
        "/home/",
        "/root/",
        "/Users/",
    )
    for item in forbidden_text:
        if item in source:
            raise ValueError(
                f"Generated renderer contains forbidden text: {item}"
            )

    return tree


def generate_renderer_source(args):
    instruction = build_renderer_instruction(
        prompt=args.prompt,
        num_frames=args.num_frames,
        fps=args.fps,
        width=args.width,
        height=args.height,
    )
    result = call_ollama(
        instruction,
        temperature=0.1,
        num_predict=7000,
    )
    response = result.get("response", "").strip()
    source = extract_python_source(response)

    if not source:
        done_reason = result.get("done_reason", "unknown")
        thinking_length = len(result.get("thinking", ""))
        raise ValueError(
            "Ollama returned no renderer code. "
            f"done_reason={done_reason}, "
            f"thinking_characters={thinking_length}"
        )

    validate_generated_renderer(source)
    return source


def run_complated_mode(args):
    inference_path = Path(args.inference_path)
    output_dir = Path(args.output_dir)

    if not args.render_only and not inference_path.exists():
        print(
            f"CogVideoX inference script not found: {inference_path}",
            file=sys.stderr,
        )
        sys.exit(1)

    if Path(args.name).name != args.name:
        print(
            "--name must be a filename stem without directories.",
            file=sys.stderr,
        )
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    original_path = output_dir / f"{args.name}_original.mp4"
    default_final_path = output_dir / f"{args.name}_final.mp4"
    final_path = (
        Path(args.final_path)
        if args.final_path
        else default_final_path
    )
    prompt_record_path = output_dir / f"{args.name}_prompts.txt"
    renderer_code_path = output_dir / f"{args.name}_generated_renderer.py"

    optimized_prompt = optimize_prompt(
        prompt=args.prompt,
        mode="complated",
        seed=args.seed,
    )

    print("\nOptimized V2V prompt:", file=sys.stderr)
    print(optimized_prompt, file=sys.stderr)

    prompt_record_path.write_text(
        (
            "Original prompt:\n"
            f"{args.prompt}\n\n"
            "Optimized V2V prompt:\n"
            f"{optimized_prompt}\n"
        ),
        encoding="utf-8",
    )

    print(
        "\nRequesting a new renderer from the Ollama API...",
        file=sys.stderr,
    )
    try:
        renderer_source = generate_renderer_source(args)
    except ValueError as error:
        print(f"Renderer generation failed: {error}", file=sys.stderr)
        sys.exit(1)

    renderer_code_path.write_text(
        renderer_source,
        encoding="utf-8",
    )

    render_command = [
        sys.executable,
        str(renderer_code_path.resolve()),
        str(original_path.resolve()),
    ]
    run_command(
        render_command,
        "Executing the API-generated renderer...",
        cwd=str(output_dir.resolve()),
        timeout=180,
    )

    if not original_path.exists():
        print(
            f"Renderer completed but did not create {original_path}.",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.render_only:
        print(f"\nReference video: {original_path}")
        print(f"Generated renderer: {renderer_code_path}")
        print(f"Prompt record: {prompt_record_path}")
        print("V2V generation was skipped because --render_only was used.")
        return

    final_path.parent.mkdir(parents=True, exist_ok=True)
    v2v_command = [
        sys.executable,
        str(inference_path),
        "--prompt",
        optimized_prompt,
        "--image_or_video_path",
        str(original_path),
        "--model_path",
        args.model_path,
        "--generate_type",
        "v2v",
        "--num_inference_steps",
        str(args.num_inference_steps),
        "--guidance_scale",
        str(args.guidance_scale),
        "--fps",
        str(args.fps),
        "--dtype",
        args.dtype,
        "--seed",
        str(args.seed),
        "--output_path",
        str(final_path),
    ]

    if args.negative_prompt:
        v2v_command.extend(
            ["--negative_prompt", args.negative_prompt]
        )

    run_command(v2v_command, "Running CogVideoX V2V...")

    print(f"\nReference video: {original_path}")
    print(f"Final video: {final_path}")
    print(f"Generated renderer: {renderer_code_path}")
    print(f"Prompt record: {prompt_record_path}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Optimize CogVideoX-5B prompts with a local Ollama model."
        )
    )

    parser.add_argument(
        "--prompt",
        required=True,
        help="Original video-generation prompt.",
    )

    parser.add_argument(
        "--mode",
        choices=["general", "bias", "complated"],
        default="general",
        help=(
            "Choose general optimization, deterministic bias mitigation, "
            "or API-generated reference rendering with V2V."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help=(
            "CogVideoX seed. In bias mode it deterministically selects "
            "one of 30 appearance profiles."
        ),
    )

    parser.add_argument(
        "--name",
        default="statue_orbit",
        help=(
            "Base filename used by complated mode. Outputs are named "
            "<name>_original.mp4 and <name>_final.mp4."
        ),
    )
    parser.add_argument(
        "--output_dir",
        default="./tests/improve/complated",
        help="Output directory used by complated mode.",
    )
    parser.add_argument(
        "--final_path",
        default=None,
        help=(
            "Optional custom final V2V output path. If omitted, "
            "<output_dir>/<name>_final.mp4 is used."
        ),
    )
    parser.add_argument("--num_frames", type=int, default=49)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--width", type=int, default=720)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument(
        "--inference_path",
        default="./inference/cli_demo.py",
        help="Path to the CogVideoX CLI inference script.",
    )
    parser.add_argument(
        "--model_path",
        default="THUDM/CogVideoX-5b",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=6.0,
    )
    parser.add_argument(
        "--dtype",
        choices=["float16", "bfloat16"],
        default="float16",
    )
    parser.add_argument(
        "--negative_prompt",
        default=None,
        help="Optional Negative Prompt passed to CogVideoX V2V.",
    )
    parser.add_argument(
        "--render_only",
        action="store_true",
        help="Render the reference video without running CogVideoX V2V.",
    )

    args = parser.parse_args()

    if args.mode == "complated":
        run_complated_mode(args)
        return

    optimized_prompt = optimize_prompt(
        prompt=args.prompt,
        mode=args.mode,
        seed=args.seed,
    )
    print(optimized_prompt)


if __name__ == "__main__":
    main()
