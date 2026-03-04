from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import h5py
import typer

app = typer.Typer(help="Print a tree view of an HDF5 file.")


def _format_attrs(attrs: dict[str, Any], max_len: int | None) -> str:
    if not attrs:
        return ""
    parts = []
    for key, value in attrs.items():
        text = f"{key}={value!r}"
        if max_len is not None and len(text) > max_len:
            text = text[: max_len - 3] + "..."
        parts.append(text)
    return " {" + ", ".join(parts) + "}"


def _matches_filter(full_path: str, name: str, flt: Optional[str]) -> bool:
    if not flt:
        return True
    flt = flt.lower()
    return flt in full_path.lower() or flt in name.lower()


def _print_group(
    group: h5py.Group,
    prefix: str,
    show_attrs: bool,
    max_attr_len: int | None,
    max_depth: int | None,
    current_depth: int,
    flt: Optional[str],
    base_path: str,
) -> None:
    if max_depth is not None and current_depth > max_depth:
        return
    keys = list(group.keys())
    for idx, name in enumerate(keys):
        is_last = idx == len(keys) - 1
        branch = "└── " if is_last else "├── "
        child = group[name]
        full_path = f"{base_path}/{name}"
        if isinstance(child, h5py.Dataset):
            shape = child.shape
            dtype = child.dtype
            attrs = dict(child.attrs) if show_attrs else {}
            attr_text = _format_attrs(attrs, max_attr_len)
            if _matches_filter(full_path, name, flt):
                print(f"{prefix}{branch}{name} {shape} {dtype}{attr_text}")
        else:
            attrs = dict(child.attrs) if show_attrs else {}
            attr_text = _format_attrs(attrs, max_attr_len)
            if _matches_filter(full_path, name, flt):
                print(f"{prefix}{branch}{name}/{attr_text}")
            extension = "    " if is_last else "│   "
            _print_group(
                child,
                prefix + extension,
                show_attrs,
                max_attr_len,
                max_depth,
                current_depth + 1,
                flt,
                full_path,
            )


@app.command()
def main(
    ctx: typer.Context,
    path: Optional[Path] = typer.Argument(None, exists=True, file_okay=True, dir_okay=False, readable=True),
    show_attrs: bool = typer.Option(False, "--attrs", help="Show dataset/group attributes."),
    max_attr_len: int | None = typer.Option(120, "--attr-max", help="Max length for each attribute entry."),
    max_depth: int | None = typer.Option(None, "--max-depth", help="Limit recursion depth."),
    flt: Optional[str] = typer.Option(None, "--filter", help="Only show paths matching this string."),
    show_metadata: bool = typer.Option(False, "--show-metadata", help="Print metadata/id and metadata/execute if present."),
) -> None:
    """Print a tree of groups/datasets with shapes and dtypes."""
    if path is None:
        print(ctx.get_help())
        raise typer.Exit(code=0)
    with h5py.File(path, "r") as h5f:
        print(f"{path}")
        _print_group(h5f, "", show_attrs, max_attr_len, max_depth, 0, flt, "")
        if show_metadata and "metadata" in h5f:
            meta = h5f["metadata"]
            for key in meta.keys():
                val = meta[key][()]
                if isinstance(val, (bytes, bytearray)):
                    val = val.decode()
                print(f"metadata/{key}: {val}")


if __name__ == "__main__":
    app()
