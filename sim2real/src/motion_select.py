# interactive_motion_sender.py
import argparse
import sys
import time
from typing import List, Tuple, Dict, Any
import yaml
import os
from pathlib import Path
from paths import SIM2REAL_ROOT, SUPPORTED_ROBOTS, tracking_config_path

SIM2REAL_SRC = SIM2REAL_ROOT / "src"
if str(SIM2REAL_SRC) not in sys.path:
    sys.path.insert(0, str(SIM2REAL_SRC))

from common.udp_latest import UDPLatestSender

BANNER = """\
Motion Sender
  - type a number or a name to send
  - press Enter to resend the last choice
  - 'list' : show all options
  - 'r'    : reload YAML
  - '?'    : help
  - 'q'    : quit
"""

def load_yaml_options(path: str) -> Tuple[List[str], Dict[str, Any]]:
    """
    Load YAML and extract unique motion names in display order.
    - motions[].name
    - motion_clips[].name  (ensures 'default' is present if defined here)
    Returns (options, raw_dict)
    """
    with open(path, "r") as f:
        data = yaml.safe_load(f) or {}

    opts: List[str] = []
    seen = set()

    # motion_clips first so 'default' appears at the top if provided there
    for arr_key in ("motion_clips", "motions"):
        arr = data.get(arr_key, []) or []
        for item in arr:
            name = str(item.get("name", "")).strip()
            if name and name not in seen:
                opts.append(name)
                seen.add(name)

    # If neither section provided default, still ensure it shows (optional)
    if "default" not in seen:
        # Not strictly necessary, but many flows expect it
        opts.insert(0, "default")
        seen.add("default")

    return opts, data

def print_menu(options: List[str]):
    print("\n=== Available motions ===")
    width = len(str(len(options)))
    for i, name in enumerate(options, 1):
        print(f"  {str(i).rjust(width)}. {name}")
    print("=========================\n")

def resolve_choice(user_in: str, options: List[str]) -> Tuple[bool, str, str]:
    """
    Resolve user input to a motion name.
    Returns (ok, resolved_name, msg)
    """
    s = user_in.strip()
    if not s:
        return False, "", "empty"

    # direct number
    if s.isdigit():
        idx = int(s)
        if 1 <= idx <= len(options):
            return True, options[idx - 1], ""
        return False, "", f"index out of range: {idx}"

    # exact match
    for name in options:
        if s == name:
            return True, name, ""

    # substring match
    matches = [n for n in options if s.lower() in n.lower()]
    if len(matches) == 1:
        return True, matches[0], ""
    elif len(matches) > 1:
        return False, "", f"ambiguous: {matches}"

    return False, "", f"unknown: '{s}'"

def send_udp(name: str, host: str, port: int, sender: UDPLatestSender) -> bool:
    try:
        sender.send({"motion": name})
        ts = time.strftime("%H:%M:%S")
        print(f"[{ts}] Sent motion '{name}' to udp://{host}:{port}")
        return True
    except Exception as e:
        print(f"[ERROR] send failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Interactive UDP motion sender")
    parser.add_argument("--robot", choices=list(SUPPORTED_ROBOTS), default="g1")
    parser.add_argument(
        "--tracking-config",
        default="tracking.yaml",
        help=(
            "Tracking YAML to read motions from. Relative names resolve under config/<robot>/; "
            "use tracking_compliance.yaml for the G1 compliance policy."
        ),
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=28562)
    args = parser.parse_args()

    yaml_path = tracking_config_path(args.robot, args.tracking_config)

    # Load options (with simple retry if file missing)
    options: List[str] = []
    raw: Dict[str, Any] = {}
    last_mtime = None
    yp = None

    def try_load():
        nonlocal options, raw, last_mtime, yp
        yp = Path(yaml_path)
        if not yp.is_absolute():
            yp = SIM2REAL_ROOT / yp
        if not os.path.exists(yp):
            print(f"[WARN] YAML not found: {yaml_path} — using only 'default'")
            options = ["default"]
            raw = {}
            last_mtime = None
            return

        options, raw = load_yaml_options(str(yp))
        try:
            last_mtime = os.path.getmtime(yp)
        except Exception:
            last_mtime = None
        print(f"[INFO] Loaded {len(options)} options from {yaml_path}")
        print_menu(options)

    try_load()

    sender = UDPLatestSender(args.host, args.port)
    print(BANNER)

    last_choice = None

    while True:
        try:
            # auto-reload on file change
            if yp and os.path.exists(yp):
                try:
                    mtime = os.path.getmtime(yp)
                    if last_mtime is not None and mtime > last_mtime:
                        try_load()
                except Exception:
                    pass

            prompt = f"Select motion [number/name | Enter:{last_choice or '-'} | list | r | ? | q]: "
            user_in = input(prompt).strip()

            if user_in == "":
                if last_choice:
                    send_udp(last_choice, args.host, args.port, sender)
                else:
                    print("Nothing to resend yet.")
                continue

            if user_in.lower() in ("q", "quit", "exit"):
                print("Bye.")
                break

            if user_in.lower() in ("?", "h", "help"):
                print(BANNER)
                continue

            if user_in.lower() in ("l", "list"):
                print_menu(options)
                continue

            if user_in.lower() == "r":
                try_load()
                continue

            ok, name, msg = resolve_choice(user_in, options)
            if not ok:
                if msg.startswith("ambiguous:"):
                    print(msg)
                else:
                    print(f"[WARN] {msg}. Type 'list' to see options.")
                continue

            if send_udp(name, args.host, args.port, sender):
                last_choice = name

        except KeyboardInterrupt:
            print("\nCtrl-C — quitting.")
            break
        except EOFError:
            print("\nEOF — quitting.")
            break
        except Exception as e:
            print(f"[ERROR] {e}")

    sender.close()

if __name__ == "__main__":
    main()
