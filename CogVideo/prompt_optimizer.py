import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import urllib.error
import urllib.request


OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
OLLAMA_MODEL = "gpt-oss:20b"


# These profiles describe visible appearance. Selection is deterministic, so
# the same CogVideoX seed always produces the same prompt.
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
    index = (
        seed * PROFILE_MULTIPLIER + PROFILE_OFFSET
    ) % len(APPEARANCE_PROFILES)
    return index, APPEARANCE_PROFILES[index]


def safe_output_name(prompt):
    words = re.findall(r"[a-z0-9]+", prompt.lower())
    prefix = "_".join(words[:5]) or "generated_video"
    prefix = prefix[:48].rstrip("_")
    digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:8]
    return f"{prefix}_{digest}"


def valid_name(name):
    return bool(
        name
        and Path(name).name == name
        and re.fullmatch(r"[a-z0-9_]+", name)
    )


def build_prompt_instruction(mode, prompt, seed=None):
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
        return f"""
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
""".strip()

    return f"""
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
""".strip()


def claude_executable(command):
    candidate = Path(command).expanduser()
    if candidate.exists() and candidate.is_file():
        return str(candidate.resolve())

    path = shutil.which(command)
    if path:
        return str(Path(path).resolve())

    raise RuntimeError(
        f"Claude Code command not found: {command}\n"
        "Install it with:\n"
        "curl -fsSL https://claude.ai/install.sh | bash"
    )


def claude_command(
    args,
    max_turns,
    tools,
    allowed_tools=None,
):
    command = [
        claude_executable(args.claude_command),
        "-p",
        "--no-session-persistence",
        "--tools",
        tools,
    ]

    if allowed_tools:
        command.append("--allowedTools")
        command.extend(allowed_tools)

    command.extend(
        [
            "--output-format",
            "text",
            "--max-turns",
            str(max_turns),
        ]
    )

    if args.claude_model:
        command.extend(["--model", args.claude_model])
    if args.claude_effort:
        command.extend(["--effort", args.claude_effort])

    return command


def call_claude(
    args,
    instruction,
    *,
    cwd=None,
    max_turns=1,
    tools="",
    allowed_tools=None,
    timeout=None,
):
    command = claude_command(
        args=args,
        max_turns=max_turns,
        tools=tools,
        allowed_tools=allowed_tools,
    )
    command.append(instruction)

    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout or args.claude_timeout,
        )
    except FileNotFoundError as error:
        raise RuntimeError(f"Unable to start Claude Code: {error}") from error
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(
            f"Claude Code exceeded the {timeout or args.claude_timeout}-"
            "second time limit."
        ) from error

    if result.returncode != 0:
        raise RuntimeError(
            "Claude Code failed.\n"
            f"exit_code={result.returncode}\n"
            f"stdout:\n{result.stdout[-4000:]}\n"
            f"stderr:\n{result.stderr[-8000:]}"
        )

    return result.stdout.strip(), result.stderr.strip()


def call_ollama(instruction, temperature=0.2, num_predict=512):
    request_data = {
        "model": OLLAMA_MODEL,
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
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Ollama HTTP error {error.code}: {body}"
        ) from error
    except urllib.error.URLError as error:
        raise RuntimeError(
            f"Unable to connect to Ollama at {OLLAMA_URL}: {error}"
        ) from error
    except (TimeoutError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Invalid Ollama response: {error}") from error


def clean_prompt_output(text):
    text = text.strip()
    fence = re.fullmatch(
        r"```(?:text)?\s*(.*?)\s*```",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if fence:
        text = fence.group(1).strip()

    text = re.sub(
        r"^(?:optimized prompt|final prompt|prompt)\s*:\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    return text.strip().strip('"').strip()


def optimize_prompt(args, mode, prompt, seed=None):
    instruction = build_prompt_instruction(mode, prompt, seed)
    if args.text_backend == "claude":
        response, _ = call_claude(
            args,
            instruction,
            max_turns=1,
            tools="",
        )
        optimized_prompt = clean_prompt_output(response)
    else:
        result = call_ollama(instruction)
        optimized_prompt = clean_prompt_output(
            result.get("response", "")
        )

    if not optimized_prompt:
        raise RuntimeError(
            f"{args.text_backend} did not return an optimized prompt."
        )

    word_count = len(optimized_prompt.split())
    print(f"Optimized prompt length: {word_count} words", file=sys.stderr)
    if word_count < 35 or word_count > 60:
        print(
            "Warning: the optimized prompt is outside the requested "
            "35-60 word range.",
            file=sys.stderr,
        )
    return optimized_prompt


def build_renderer_instruction(
    prompt,
    script_name,
    video_name,
    check_name,
    num_frames,
    fps,
    width,
    height,
    previous_error=None,
):
    repair_section = ""
    if previous_error:
        repair_section = f"""

PREVIOUS ATTEMPT PROBLEM:
{previous_error}

Inspect the existing files, repair them, rerun the renderer, and verify the
new result. Do not merely explain the error.
"""

    return f"""
Work only in the current directory. Create and execute a deterministic,
self-contained Python reference-animation renderer for the user request
below.

USER VIDEO REQUEST:
{prompt}

REQUIRED OUTPUTS:
- Renderer source: {script_name}
- Reference MP4: {video_name}
- Visual contact sheet: {check_name}

VIDEO SETTINGS:
- Frames: {num_frames}
- FPS: {fps}
- Resolution: {width}x{height}

PURPOSE:
The MP4 will guide a later CogVideoX video-to-video pass. Exact subject
motion, camera motion, action order, spatial relationships, timing, direction,
and requested rotation count matter more than photorealism.

RENDERER REQUIREMENTS:
- Create a simple but clearly recognizable symbolic animation.
- Use Python with argparse, math, pathlib, NumPy, Pillow, and imageio_ffmpeg.
- Do not use Blender, Manim, OpenGL, external assets, downloads, or network
  access.
- Use explicit mathematical trajectories and chronological keyframes.
- Make different sides of important subjects visually distinguishable.
- For a camera orbit, move the virtual camera around a stationary subject;
  do not rotate the subject to face the camera.
- Complete exact requested rotation counts mathematically.
- If the ending viewpoint should equal the beginning, make the first and last
  camera states mathematically identical.
- Keep object identity, size, color, and stationary objects consistent.
- Avoid morphing, unexplained cuts, duplicates, or extra subjects.
- Encode a real H.264 MP4 with yuv420p pixel format.
- imageio_ffmpeg.write_frames returns a generator. Initialize it, call
  writer.send(None), send contiguous uint8 RGB frames one by one, and close it
  in a finally block.

VALIDATION REQUIREMENTS:
- Run it using exactly: python {script_name}
- Confirm {video_name} exists, is non-empty, and is readable.
- Build {check_name} from several frames covering the entire motion.
- Use the Read tool to inspect the contact sheet.
- If the requested motion is not visually clear or the output is invalid,
  modify the renderer and repeat the test.
- Finish only after the three required files exist.

Do not run CogVideoX. Do not edit files outside the current directory.
In the final response, report only whether validation succeeded and list the
three filenames.
{repair_section}
""".strip()


def validate_python_source(path):
    if not path.exists() or path.stat().st_size == 0:
        raise RuntimeError(f"Generated renderer is missing: {path.name}")
    try:
        compile(
            path.read_text(encoding="utf-8"),
            str(path),
            "exec",
        )
    except (OSError, UnicodeDecodeError, SyntaxError) as error:
        raise RuntimeError(
            f"Generated renderer is not valid Python: {error}"
        ) from error


def probe_video(path):
    if not path.exists() or path.stat().st_size < 1024:
        raise RuntimeError(
            f"Reference video is missing or too small: {path.name}"
        )

    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        print(
            "Warning: ffprobe was not found; only file size was checked.",
            file=sys.stderr,
        )
        return None

    result = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,width,height,nb_frames,r_frame_rate",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "ffprobe could not read the reference video.\n"
            f"{result.stderr[-4000:]}"
        )

    try:
        metadata = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError(
            f"ffprobe returned invalid JSON: {error}"
        ) from error

    streams = metadata.get("streams", [])
    if not streams:
        raise RuntimeError("Reference MP4 contains no video stream.")
    return streams[0]


def generate_claude_reference(args, output_dir, name):
    script_path = output_dir / f"{name}_generated_renderer.py"
    original_path = output_dir / f"{name}_original.mp4"
    check_path = output_dir / f"{name}_check.jpg"
    task_path = output_dir / f"{name}_claude_task.txt"
    log_path = output_dir / f"{name}_claude_log.txt"

    previous_error = None
    attempts = []
    for attempt in range(args.claude_repairs + 1):
        instruction = build_renderer_instruction(
            prompt=args.prompt,
            script_name=script_path.name,
            video_name=original_path.name,
            check_name=check_path.name,
            num_frames=args.num_frames,
            fps=args.fps,
            width=args.width,
            height=args.height,
            previous_error=previous_error,
        )
        task_path.write_text(instruction + "\n", encoding="utf-8")

        print(
            f"Running Claude reference renderer "
            f"(attempt {attempt + 1}/{args.claude_repairs + 1})...",
            file=sys.stderr,
        )
        try:
            stdout, stderr = call_claude(
                args,
                instruction,
                cwd=output_dir,
                max_turns=args.claude_turns,
                tools="Bash,Edit,Read,Write",
                allowed_tools=[
                    "Read",
                    "Write",
                    "Edit",
                    "Bash(python *)",
                    "Bash(python3 *)",
                    "Bash(ffmpeg *)",
                    "Bash(ffprobe *)",
                    "Bash(file *)",
                    "Bash(ls *)",
                    "Bash(pwd)",
                ],
                timeout=args.renderer_timeout,
            )
            validate_python_source(script_path)
            metadata = probe_video(original_path)
            if not check_path.exists() or check_path.stat().st_size == 0:
                raise RuntimeError(
                    f"Visual contact sheet is missing: {check_path.name}"
                )
            attempts.append(
                {
                    "attempt": attempt + 1,
                    "status": "success",
                    "stdout": stdout[-4000:],
                    "stderr": stderr[-4000:],
                }
            )
            log_path.write_text(
                stdout + ("\n\nSTDERR:\n" + stderr if stderr else "") + "\n",
                encoding="utf-8",
            )
            return {
                "script_path": script_path,
                "original_path": original_path,
                "check_path": check_path,
                "task_path": task_path,
                "log_path": log_path,
                "video_metadata": metadata,
                "attempts": attempts,
            }
        except RuntimeError as error:
            previous_error = str(error)
            attempts.append(
                {
                    "attempt": attempt + 1,
                    "status": "failed",
                    "error": previous_error,
                }
            )
            if attempt >= args.claude_repairs:
                raise RuntimeError(
                    "Claude could not create a valid reference video after "
                    f"{attempt + 1} attempt(s).\n{previous_error}"
                ) from error
            print(
                "Reference validation failed; asking Claude to repair it...",
                file=sys.stderr,
            )

    raise RuntimeError("Reference generation ended unexpectedly.")


def make_v2v_prompt(args):
    if args.v2v_prompt:
        return args.v2v_prompt

    if args.skip_v2v_prompt_optimization:
        optimized = args.prompt
    else:
        optimized = optimize_prompt(
            args,
            mode="normal",
            prompt=args.prompt,
            seed=args.seed,
        )

    return (
        f"{optimized} Preserve the reference video's exact camera path, "
        "subject motion, action order, direction, timing, duration, rotation "
        "count, and number of subjects. Keep every subject visually "
        "consistent throughout."
    )


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
        raise RuntimeError(f"Unable to start command: {error}") from error
    except subprocess.CalledProcessError as error:
        raise RuntimeError(
            f"{description} failed with exit code {error.returncode}."
        ) from error
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(
            f"{description} exceeded the {timeout}-second time limit."
        ) from error


def add_extra_cli_args(command, values):
    for value in values:
        if "=" in value:
            name, argument = value.split("=", 1)
        else:
            name, argument = value, None

        if not re.fullmatch(r"[a-zA-Z0-9_]+", name):
            raise RuntimeError(
                f"Invalid --v2v_extra option name: {name}"
            )
        command.append(f"--{name}")
        if argument is not None:
            command.append(argument)


def build_v2v_command(args, original_path, final_path, v2v_prompt):
    command = [
        sys.executable,
        str(Path(args.inference_path)),
        "--prompt",
        v2v_prompt,
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
        command.extend(["--negative_prompt", args.negative_prompt])
    add_extra_cli_args(command, args.v2v_extra)
    return command


def run_complated_mode(args):
    output_dir = Path(args.output_dir).expanduser().resolve()
    inference_path = Path(args.inference_path).expanduser()
    if not inference_path.is_absolute():
        inference_path = (Path.cwd() / inference_path).resolve()
    args.inference_path = str(inference_path)

    name = args.name or safe_output_name(args.prompt)
    if not valid_name(name):
        raise RuntimeError(
            "--name may contain only lowercase letters, digits, and "
            "underscores."
        )

    if args.num_frames < 2:
        raise RuntimeError("--num_frames must be at least 2.")
    if args.fps < 1 or args.width < 1 or args.height < 1:
        raise RuntimeError(
            "--fps, --width, and --height must be positive."
        )
    if args.claude_turns < 1 or args.claude_repairs < 0:
        raise RuntimeError(
            "--claude_turns must be positive and --claude_repairs cannot "
            "be negative."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    original_path = output_dir / f"{name}_original.mp4"
    final_path = (
        Path(args.final_path).expanduser().resolve()
        if args.final_path
        else output_dir / f"{name}_final.mp4"
    )
    record_path = output_dir / f"{name}_generation.json"

    if not args.reference_only and not inference_path.exists():
        raise RuntimeError(
            f"CogVideoX inference script not found: {inference_path}"
        )

    if args.dry_run:
        v2v_prompt = args.v2v_prompt or args.prompt
        command = build_v2v_command(
            args,
            original_path,
            final_path,
            v2v_prompt,
        )
        print("Dry run: no Claude or CogVideoX command was executed.")
        print(f"Reference video: {original_path}")
        print(f"Final video: {final_path}")
        print("Planned V2V command:")
        print(" ".join(command))
        return

    reference = generate_claude_reference(args, output_dir, name)
    if args.reference_only:
        v2v_prompt = args.v2v_prompt or args.prompt
    else:
        v2v_prompt = make_v2v_prompt(args)

    record = {
        "mode": "complated",
        "original_prompt": args.prompt,
        "text_backend": args.text_backend,
        "claude_model": args.claude_model,
        "reference_renderer": str(reference["script_path"]),
        "reference_video": str(reference["original_path"]),
        "reference_contact_sheet": str(reference["check_path"]),
        "reference_video_metadata": reference["video_metadata"],
        "reference_attempts": reference["attempts"],
        "num_frames": args.num_frames,
        "fps": args.fps,
        "width": args.width,
        "height": args.height,
        "v2v_prompt": v2v_prompt,
        "negative_prompt": args.negative_prompt,
        "final_video": None if args.reference_only else str(final_path),
    }
    record_path.write_text(
        json.dumps(record, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    if args.reference_only:
        print(f"Reference video: {reference['original_path']}")
        print(f"Contact sheet: {reference['check_path']}")
        print(f"Renderer: {reference['script_path']}")
        print(f"Generation record: {record_path}")
        return

    final_path.parent.mkdir(parents=True, exist_ok=True)
    v2v_command = build_v2v_command(
        args,
        reference["original_path"],
        final_path,
        v2v_prompt,
    )
    run_command(
        v2v_command,
        "Running CogVideoX V2V...",
        timeout=args.v2v_timeout,
    )

    if not final_path.exists() or final_path.stat().st_size == 0:
        raise RuntimeError(
            f"CogVideoX finished but did not create: {final_path}"
        )

    print(f"Reference video: {reference['original_path']}")
    print(f"Contact sheet: {reference['check_path']}")
    print(f"Final video: {final_path}")
    print(f"Generation record: {record_path}")


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Optimize CogVideoX prompts with Claude Code, apply deterministic "
            "bias mitigation, or generate a Claude-rendered reference video "
            "followed by CogVideoX V2V."
        )
    )
    parser.add_argument(
        "--prompt",
        required=True,
        help="Original video-generation prompt.",
    )
    parser.add_argument(
        "--mode",
        choices=["normal", "general", "bias", "complated"],
        default="normal",
        help=(
            "normal/general: optimize and print a prompt; bias: select a "
            "deterministic appearance profile and print a prompt; complated: "
            "render a reference animation and run CogVideoX V2V."
        ),
    )
    parser.add_argument(
        "--text_backend",
        choices=["claude", "ollama"],
        default="claude",
        help=(
            "Text optimizer. Claude is the default; Ollama remains available "
            "as an offline fallback."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help=(
            "CogVideoX seed. Bias mode also uses it to select one of 30 "
            "appearance profiles."
        ),
    )

    claude = parser.add_argument_group("Claude Code")
    claude.add_argument(
        "--claude_command",
        default="claude",
        help="Claude Code executable name or path.",
    )
    claude.add_argument(
        "--claude_model",
        default=None,
        help=(
            "Optional Claude model alias or ID. If omitted, Claude Code uses "
            "the account default."
        ),
    )
    claude.add_argument(
        "--claude_effort",
        choices=["low", "medium", "high", "xhigh", "max"],
        default=None,
        help="Optional Claude Code effort level.",
    )
    claude.add_argument(
        "--claude_timeout",
        type=int,
        default=300,
        help="Timeout for a prompt-only Claude request.",
    )
    claude.add_argument(
        "--claude_turns",
        type=int,
        default=12,
        help="Maximum agentic turns for reference rendering.",
    )
    claude.add_argument(
        "--claude_repairs",
        type=int,
        default=1,
        help=(
            "Additional Claude attempts if reference validation fails."
        ),
    )
    claude.add_argument(
        "--renderer_timeout",
        type=int,
        default=1200,
        help="Timeout for each Claude reference-rendering attempt.",
    )

    reference = parser.add_argument_group("Reference animation")
    reference.add_argument(
        "--name",
        default=None,
        help=(
            "Output prefix for complated mode. If omitted, a deterministic "
            "name is derived from the prompt."
        ),
    )
    reference.add_argument(
        "--output_dir",
        default="./tests/improve/complated",
    )
    reference.add_argument("--num_frames", type=int, default=49)
    reference.add_argument("--fps", type=int, default=8)
    reference.add_argument("--width", type=int, default=720)
    reference.add_argument("--height", type=int, default=480)
    reference.add_argument(
        "--reference_only",
        action="store_true",
        help="Create and validate only <name>_original.mp4.",
    )

    v2v = parser.add_argument_group("CogVideoX V2V")
    v2v.add_argument(
        "--final_path",
        default=None,
        help="Optional custom final-video path.",
    )
    v2v.add_argument(
        "--v2v_prompt",
        default=None,
        help="Optional CogVideoX V2V prompt override.",
    )
    v2v.add_argument(
        "--skip_v2v_prompt_optimization",
        action="store_true",
        help="Use the original prompt for V2V without another Claude call.",
    )
    v2v.add_argument(
        "--inference_path",
        default="./inference/cli_demo.py",
    )
    v2v.add_argument(
        "--model_path",
        default="THUDM/CogVideoX-5b",
    )
    v2v.add_argument(
        "--num_inference_steps",
        type=int,
        default=50,
    )
    v2v.add_argument(
        "--guidance_scale",
        type=float,
        default=6.0,
    )
    v2v.add_argument(
        "--dtype",
        choices=["float16", "bfloat16"],
        default="float16",
    )
    v2v.add_argument(
        "--negative_prompt",
        default=None,
        help="Optional Negative Prompt passed to CogVideoX V2V.",
    )
    v2v.add_argument(
        "--v2v_extra",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help=(
            "Pass an additional argument to cli_demo.py. May be repeated. "
            "Use NAME for a flag or NAME=VALUE for an option."
        ),
    )
    v2v.add_argument(
        "--v2v_timeout",
        type=int,
        default=None,
        help="Optional CogVideoX V2V timeout in seconds.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print the planned complated paths and V2V command only.",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.mode == "general":
        args.mode = "normal"

    try:
        if args.mode == "complated":
            run_complated_mode(args)
            return

        print(
            optimize_prompt(
                args,
                mode=args.mode,
                prompt=args.prompt,
                seed=args.seed,
            )
        )
    except RuntimeError as error:
        print(error, file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
