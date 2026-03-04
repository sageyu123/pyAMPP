#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

from scipy.io import readsav


_MONTHS = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}


@dataclass
class ParseResult:
    command: List[str]
    mapped: Dict[str, str]
    unmapped: Dict[str, str]
    source_execute: str


def _decode_scalar(value) -> str:
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    return str(value)


def _load_execute_from_sav(path: Path) -> str:
    data = readsav(str(path), python_dict=True, verbose=False)
    if "box" in data:
        box = data["box"][0]
    elif "pbox" in data:
        box = data["pbox"][0]
    else:
        raise KeyError(f"No 'box' or 'pbox' structure found in {path}")

    if "EXECUTE" not in box.dtype.names:
        raise KeyError(f"Missing EXECUTE in SAV box structure: {path}")
    return _decode_scalar(box["EXECUTE"]).strip()


def _split_top_level_csv(text: str) -> List[str]:
    parts: List[str] = []
    buf: List[str] = []
    depth = 0
    in_quote = False
    quote_char = ""
    for ch in text:
        if in_quote:
            buf.append(ch)
            if ch == quote_char:
                in_quote = False
            continue
        if ch in ("'", '"'):
            in_quote = True
            quote_char = ch
            buf.append(ch)
            continue
        if ch == "[":
            depth += 1
            buf.append(ch)
            continue
        if ch == "]":
            depth = max(depth - 1, 0)
            buf.append(ch)
            continue
        if ch == "," and depth == 0:
            token = "".join(buf).strip()
            if token:
                parts.append(token)
            buf = []
            continue
        buf.append(ch)
    token = "".join(buf).strip()
    if token:
        parts.append(token)
    return parts


def _strip_quotes(text: str) -> str:
    t = text.strip()
    if len(t) >= 2 and ((t[0] == "'" and t[-1] == "'") or (t[0] == '"' and t[-1] == '"')):
        return t[1:-1]
    return t


def _parse_idl_time(text: str) -> str:
    s = _strip_quotes(text).strip()
    for fmt in ("%d-%b-%y %H:%M:%S", "%d-%b-%Y %H:%M:%S"):
        try:
            dt = datetime.strptime(s, fmt)
            return dt.strftime("%Y-%m-%dT%H:%M:%S")
        except ValueError:
            pass

    # Manual fallback to keep parser robust for abbreviated/variant month casing.
    m = re.match(r"^\s*(\d{1,2})-([A-Za-z]{3})-(\d{2,4})\s+(\d{2}):(\d{2}):(\d{2})\s*$", s)
    if not m:
        return s
    day, mon_s, year_s, hh, mm, ss = m.groups()
    mon = _MONTHS.get(mon_s.lower())
    if mon is None:
        return s
    year = int(year_s)
    if year < 100:
        year += 2000 if year < 70 else 1900
    dt = datetime(year, mon, int(day), int(hh), int(mm), int(ss))
    return dt.strftime("%Y-%m-%dT%H:%M:%S")


def _parse_vector(value: str) -> List[str]:
    inner = value.strip()
    if inner.startswith("[") and inner.endswith("]"):
        inner = inner[1:-1]
    return [x.strip() for x in inner.split(",") if x.strip()]


def _parse_idl_call(execute: str) -> Tuple[str, Dict[str, str]]:
    cleaned = " ".join(execute.replace("\n", " ").split())
    if "," not in cleaned:
        # already likely a plain command, keep as-is with no keyword map
        return cleaned, {}
    head, tail = cleaned.split(",", 1)
    func = head.strip()
    tokens = _split_top_level_csv(tail)

    kw: Dict[str, str] = {}
    # First positional argument is expected to be time string.
    if tokens and "=" not in tokens[0]:
        kw["TIME"] = _strip_quotes(tokens[0])
        tokens = tokens[1:]
    for tok in tokens:
        if "=" not in tok:
            continue
        k, v = tok.split("=", 1)
        kw[k.strip().upper()] = v.strip()
    return func, kw


def _is_truthy(v: str) -> bool:
    t = v.strip().lower()
    return t in {"1", "true", "t", "yes", "y"}


def _build_gx_fov2box_command(kw: Dict[str, str]) -> ParseResult:
    cmd: List[str] = ["gx-fov2box"]
    mapped: Dict[str, str] = {}
    unmapped: Dict[str, str] = {}

    # Required core fields
    if "TIME" in kw:
        iso = _parse_idl_time(kw["TIME"])
        cmd += ["--time", iso]
        mapped["TIME"] = iso
    if "CENTER_ARCSEC" in kw:
        vv = _parse_vector(kw["CENTER_ARCSEC"])
        if len(vv) >= 2:
            cmd += ["--coords", vv[0], vv[1], "--hpc"]
            mapped["CENTER_ARCSEC"] = f"{vv[0]} {vv[1]} (HPC)"
    if "SIZE_PIX" in kw:
        vv = _parse_vector(kw["SIZE_PIX"])
        if len(vv) >= 3:
            cmd += ["--box-dims", vv[0], vv[1], vv[2]]
            mapped["SIZE_PIX"] = " ".join(vv[:3])
    if "DX_KM" in kw:
        cmd += ["--dx-km", _strip_quotes(kw["DX_KM"])]
        mapped["DX_KM"] = _strip_quotes(kw["DX_KM"])

    # Direct path mappings
    direct_map = {
        "TMP_DIR": "--data-dir",
        "OUT_DIR": "--gxmodel-dir",
        "PAD_FRAC": "--pad-frac",
    }
    for k, flag in direct_map.items():
        if k in kw:
            val = _strip_quotes(kw[k])
            cmd += [flag, val]
            mapped[k] = val

    # Projection / toggles
    if "CEA" in kw and _is_truthy(kw["CEA"]):
        cmd.append("--cea")
        mapped["CEA"] = "1"
    if "TOP" in kw and _is_truthy(kw["TOP"]):
        cmd.append("--top")
        mapped["TOP"] = "1"
    if "EUV" in kw and _is_truthy(kw["EUV"]):
        cmd.append("--euv")
        mapped["EUV"] = "1"
    if "UV" in kw and _is_truthy(kw["UV"]):
        cmd.append("--uv")
        mapped["UV"] = "1"
    if "SFQ" in kw and _is_truthy(kw["SFQ"]):
        cmd.append("--sfq")
        mapped["SFQ"] = "1"

    save_flags = {
        "SAVE_EMPTY_BOX": "--save-empty-box",
        "SAVE_POTENTIAL": "--save-potential",
        "SAVE_BOUNDS": "--save-bounds",
        "SAVE_NAS": "--save-nas",
        "SAVE_GEN": "--save-gen",
        "SAVE_CHR": "--save-chr",
    }
    for k, flag in save_flags.items():
        if k in kw and _is_truthy(kw[k]):
            cmd.append(flag)
            mapped[k] = "1"

    # Stage controls
    if "POTENTIAL_ONLY" in kw and _is_truthy(kw["POTENTIAL_ONLY"]):
        cmd.append("--potential-only")
        mapped["POTENTIAL_ONLY"] = "1"
    if "NLFFF_ONLY" in kw and _is_truthy(kw["NLFFF_ONLY"]):
        cmd.append("--nlfff-only")
        mapped["NLFFF_ONLY"] = "1"
    if "GENERIC_ONLY" in kw and _is_truthy(kw["GENERIC_ONLY"]):
        cmd.append("--generic-only")
        mapped["GENERIC_ONLY"] = "1"
    if "EMPTY_BOX_ONLY" in kw and _is_truthy(kw["EMPTY_BOX_ONLY"]):
        cmd.append("--empty-box-only")
        mapped["EMPTY_BOX_ONLY"] = "1"

    known = set(mapped.keys())
    for k, v in kw.items():
        if k not in known:
            unmapped[k] = _strip_quotes(v)

    return ParseResult(command=cmd, mapped=mapped, unmapped=unmapped, source_execute="")


def _override_flag_value(cmd: List[str], flag: str, value: str) -> List[str]:
    out: List[str] = []
    i = 0
    replaced = False
    while i < len(cmd):
        if cmd[i] == flag:
            out.extend([flag, value])
            replaced = True
            i += 2
            continue
        out.append(cmd[i])
        i += 1
    if not replaced:
        out.extend([flag, value])
    return out


def _path_warnings(path_value: str, role: str) -> List[str]:
    warnings: List[str] = []
    pv = path_value.strip()
    if not pv:
        warnings.append(f"{role}: empty path.")
        return warnings

    # Simple portability hint: Windows-style absolute path on POSIX.
    if os.name != "nt" and re.match(r"^[A-Za-z]:[\\/]", pv):
        warnings.append(f"{role}: looks like Windows path on POSIX host: {pv}")
        return warnings
    # POSIX-style absolute path on Windows.
    if os.name == "nt" and pv.startswith("/"):
        warnings.append(f"{role}: looks like POSIX path on Windows host: {pv}")
        return warnings

    p = Path(pv).expanduser()
    if p.exists():
        return warnings
    warnings.append(f"{role}: path does not exist: {p}")
    warnings.append(f"{role}: gx-fov2box will create it if parent is writable.")
    return warnings


def _format_multiline_command(cmd: List[str]) -> str:
    if len(cmd) <= 3:
        return " ".join(cmd)
    out = [cmd[0] + " \\"]
    i = 1
    arity = {
        "--coords": 2,
        "--box-dims": 3,
    }
    while i < len(cmd):
        if cmd[i].startswith("--"):
            n = arity.get(cmd[i], 1)
            vals: List[str] = []
            j = i + 1
            while j < len(cmd) and len(vals) < n and not cmd[j].startswith("--"):
                vals.append(cmd[j])
                j += 1
            if vals:
                out.append(f"  {cmd[i]} {' '.join(vals)} \\")
                i = j
                continue
            # Boolean flag
            out.append(f"  {cmd[i]} \\")
            i += 1
        else:
            sep = " \\" if i < len(cmd) - 1 else ""
            out.append(f"  {cmd[i]}{sep}")
            i += 1
    # Remove trailing slash from final line.
    out[-1] = out[-1].rstrip(" \\")
    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Translate IDL gx_fov2box EXECUTE strings into equivalent gx-fov2box CLI commands."
    )
    parser.add_argument("--execute", type=str, help="Raw IDL execute string (e.g., gx_fov2box, '26-Nov-25 ...').")
    parser.add_argument("--sav", type=Path, help="Path to IDL SAV model; reads box/pbox EXECUTE.")
    parser.add_argument("--out-script", type=Path, help="Optional output .sh script path.")
    parser.add_argument("--out-json", type=Path, help="Optional output JSON report path.")
    parser.add_argument(
        "--data-dir",
        type=str,
        help="Override translated --data-dir path (TMP_DIR) for portability.",
    )
    parser.add_argument(
        "--gxmodel-dir",
        type=str,
        help="Override translated --gxmodel-dir path (OUT_DIR) for portability.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print both original IDL execute string and translated Python command for visual comparison.",
    )
    args = parser.parse_args()

    if not args.execute and not args.sav:
        parser.error("Provide one of --execute or --sav.")
    if args.execute and args.sav:
        parser.error("Use either --execute or --sav, not both.")

    if args.sav:
        execute = _load_execute_from_sav(args.sav)
        source = str(args.sav)
    else:
        execute = args.execute or ""
        source = "cli"

    _, kw = _parse_idl_call(execute)
    result = _build_gx_fov2box_command(kw)
    result.source_execute = execute

    if args.data_dir:
        result.command = _override_flag_value(result.command, "--data-dir", args.data_dir)
        result.mapped["TMP_DIR"] = args.data_dir
    if args.gxmodel_dir:
        result.command = _override_flag_value(result.command, "--gxmodel-dir", args.gxmodel_dir)
        result.mapped["OUT_DIR"] = args.gxmodel_dir

    warnings: List[str] = []
    if "--data-dir" in result.command:
        i = result.command.index("--data-dir")
        if i + 1 < len(result.command):
            warnings.extend(_path_warnings(result.command[i + 1], "data-dir"))
    if "--gxmodel-dir" in result.command:
        i = result.command.index("--gxmodel-dir")
        if i + 1 < len(result.command):
            warnings.extend(_path_warnings(result.command[i + 1], "gxmodel-dir"))

    cmd_text = _format_multiline_command(result.command)
    if args.verbose:
        print("IDL execute:")
        print(result.source_execute)
        print()
        print("Translated gx-fov2box:")
        print(cmd_text)
        print()
    else:
        print(cmd_text)
    if warnings:
        for w in warnings:
            print(f"WARNING: {w}", file=sys.stderr)

    report = {
        "source": source,
        "execute": execute,
        "translated_command": result.command,
        "mapped_keywords": result.mapped,
        "unmapped_keywords": result.unmapped,
        "warnings": warnings,
    }

    if args.out_script:
        args.out_script.parent.mkdir(parents=True, exist_ok=True)
        args.out_script.write_text("#!/usr/bin/env bash\nset -euo pipefail\n\n" + cmd_text + "\n", encoding="utf-8")
        try:
            args.out_script.chmod(0o755)
        except OSError:
            pass

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
