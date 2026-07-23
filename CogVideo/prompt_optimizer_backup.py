import argparse
import json
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


def call_ollama(instruction):
    request_data = {
        "model": MODEL,
        "prompt": instruction,
        "stream": False,
        "think": "low",
        "keep_alive": 0,
        "options": {
            "temperature": 0.2,
            "num_predict": 512,
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

    if word_count < 35 or word_count > 60:
        print(
            "Warning: the optimized prompt is outside the requested "
            "35-60 word range.",
            file=sys.stderr,
        )

    return optimized_prompt


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
        choices=["general", "bias"],
        default="general",
        help="Choose general optimization or deterministic bias mitigation.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        help=(
            "CogVideoX seed. In bias mode it deterministically selects "
            "one of 30 appearance profiles."
        ),
    )

    args = parser.parse_args()

    if args.mode == "bias" and args.seed is None:
        parser.error("--seed is required when --mode bias is selected.")

    optimized_prompt = optimize_prompt(
        prompt=args.prompt,
        mode=args.mode,
        seed=args.seed,
    )
    print(optimized_prompt)


if __name__ == "__main__":
    main()
