import shlex
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import astropy.units as u
import numpy as np
import typer
from astropy.coordinates import SkyCoord
from astropy.time import Time
from astropy.io import fits
from pyAMaFiL.mag_field_lin_fff import MagFieldLinFFF
from pyAMaFiL.mag_field_proc import MagFieldProcessor
from sunpy.coordinates import (Heliocentric, HeliographicCarrington, HeliographicStonyhurst,
                               Helioprojective, get_earth)
from sunpy.map import Map
from sunpy.sun import constants as sun_consts

from pyampp.data.downloader import SDOImageDownloader
from pyampp.gx_chromo.combo_model import combo_model
from pyampp.gx_chromo.decompose import decompose
from pyampp.gxbox.box import Box
from pyampp.gxbox.boxutils import hmi_b2ptr, hmi_disambig, read_b3d_h5, write_b3d_h5, compute_vertical_current
from pyampp.gxbox.gx_box2id import gx_box2id
from pyampp.util.config import DOWNLOAD_DIR, GXMODEL_DIR, AIA_EUV_PASSBANDS, AIA_UV_PASSBANDS


app = typer.Typer(help="Run gx_fov2box pipeline without GUI and save model stages.")


@dataclass
class Fov2BoxConfig:
    time: Optional[str]
    coords: Optional[Tuple[float, float]]
    hpc: bool
    hgc: bool
    hgs: bool
    cea: bool
    top: bool
    box_dims: Optional[Tuple[int, int, int]]
    dx_km: float
    pad_frac: float
    data_dir: str
    gxmodel_dir: str
    entry_box: Optional[str]
    save_empty_box: bool
    save_potential: bool
    save_bounds: bool
    save_nas: bool
    save_gen: bool
    save_chr: bool
    stop_after: Optional[str]
    empty_box_only: bool
    potential_only: bool
    nlfff_only: bool
    generic_only: bool
    use_potential: bool
    center_vox: bool
    reduce_passed: Optional[int]
    euv: bool
    uv: bool
    sfq: bool
    jump2potential: bool
    jump2nlfff: bool
    jump2lines: bool
    jump2chromo: bool
    info: bool


def _infer_time_from_path(path: Path) -> Optional[str]:
    stem = path.name
    if "hmi.M_720s." in stem:
        try:
            tag = stem.split("hmi.M_720s.", 1)[1].split(".", 1)[0]
            if len(tag) == 15:
                return f"{tag[0:4]}-{tag[4:6]}-{tag[6:8]}T{tag[9:11]}:{tag[11:13]}:{tag[13:15]}"
        except Exception:
            return None
    return None


def _format_coord_tag(lon_deg: float, lat_deg: float, suffix: str = "CR") -> str:
    lon = (lon_deg + 180.0) % 360.0 - 180.0
    lon_dir = "W" if lon >= 0 else "E"
    lat_dir = "N" if lat_deg >= 0 else "S"
    lon_val = f"{abs(int(round(lon))):02d}"
    lat_val = f"{abs(int(round(lat_deg))):02d}"
    return f"{lon_dir}{lon_val}{lat_dir}{lat_val}{suffix}"


def _stage_output_dir(gxmodel_dir: str, obs_time: Time) -> Path:
    date_str = obs_time.to_datetime().strftime("%Y-%m-%d")
    out_dir = Path(gxmodel_dir) / date_str
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _stage_file_base(obs_time: Time, coord_tag: str, projection_tag: str = "CEA") -> str:
    time_tag = obs_time.to_datetime().strftime("%Y%m%d_%H%M%S")
    return f"hmi.M_720s.{time_tag}.{coord_tag}.{projection_tag}"


def _stage_filename(out_dir: Path, base: str, stage_tag: str) -> Path:
    return out_dir / f"{base}.{stage_tag}.h5"


def _last_stage_tag(stop_after: Optional[str]) -> str:
    if stop_after:
        stop = stop_after.lower()
        if stop in ("none", "empty", "empty_box"):
            return "NONE"
        if stop in ("bnd", "bounds"):
            return "BND"
        if stop == "pot":
            return "POT"
        if stop == "nas":
            return "NAS"
        if stop == "gen":
            return "NAS.GEN"
        if stop == "chr":
            return "NAS.CHR"
    return "NAS.CHR"


def _should_save_stage(stage_tag: str, cfg: Fov2BoxConfig) -> bool:
    if stage_tag == _last_stage_tag(cfg.stop_after):
        return True
    if stage_tag == "POT":
        return cfg.save_potential
    if stage_tag == "NAS":
        return cfg.save_nas
    if stage_tag == "NAS.GEN":
        return cfg.save_gen
    if stage_tag == "NAS.CHR":
        return cfg.save_chr
    if stage_tag == "NONE":
        return cfg.save_empty_box
    if stage_tag == "BND":
        return cfg.save_bounds
    return False


def _build_execute_cmd(cfg: Fov2BoxConfig) -> str:
    cmd = ["gx-fov2box"]
    if cfg.time:
        cmd += ["--time", cfg.time]
    if cfg.coords:
        cmd += ["--coords", str(cfg.coords[0]), str(cfg.coords[1])]
    if cfg.hpc:
        cmd.append("--hpc")
    elif cfg.hgc:
        cmd.append("--hgc")
    else:
        cmd.append("--hgs")
    if cfg.top:
        cmd.append("--top")
    elif cfg.cea:
        cmd.append("--cea")
    if cfg.box_dims:
        cmd += ["--box-dims", *(str(v) for v in cfg.box_dims)]
    cmd += ["--dx-km", f"{cfg.dx_km:.6f}"]
    cmd += ["--pad-frac", f"{cfg.pad_frac:.4f}"]
    cmd += ["--data-dir", cfg.data_dir]
    cmd += ["--gxmodel-dir", cfg.gxmodel_dir]
    cmd += ["--euv" if cfg.euv else "--no-euv"]
    cmd += ["--uv" if cfg.uv else "--no-uv"]
    if cfg.save_empty_box:
        cmd.append("--save-empty-box")
    if cfg.save_potential:
        cmd.append("--save-potential")
    if cfg.save_bounds:
        cmd.append("--save-bounds")
    if cfg.save_nas:
        cmd.append("--save-nas")
    if cfg.save_gen:
        cmd.append("--save-gen")
    if cfg.save_chr:
        cmd.append("--save-chr")
    if cfg.empty_box_only:
        cmd.append("--empty-box-only")
    if cfg.potential_only:
        cmd.append("--potential-only")
    if cfg.nlfff_only:
        cmd.append("--nlfff-only")
    if cfg.generic_only:
        cmd.append("--generic-only")
    if cfg.use_potential:
        cmd.append("--use-potential")
    if cfg.center_vox:
        cmd.append("--center-vox")
    if cfg.reduce_passed is not None:
        cmd += ["--reduce-passed", str(cfg.reduce_passed)]
    if cfg.entry_box:
        cmd += ["--entry-box", cfg.entry_box]
    if cfg.sfq:
        cmd.append("--sfq")
    if cfg.stop_after:
        cmd += ["--stop-after", cfg.stop_after]
    if cfg.jump2potential:
        cmd.append("--jump2potential")
    if cfg.jump2nlfff:
        cmd.append("--jump2nlfff")
    if cfg.jump2lines:
        cmd.append("--jump2lines")
    if cfg.jump2chromo:
        cmd.append("--jump2chromo")
    return shlex.join(cmd)


def _build_index_header(bottom_wcs_header, source_map: Map) -> str:
    """
    Build an IDL-GX compatible INDEX-like FITS header string.

    This mirrors the intent of IDL `wcs2fitshead(..., /structure)` for the
    base map WCS: preserve WCS cards and add key TIME/POSITION cards expected
    by legacy GX Simulator routines.
    """
    header = fits.Header(bottom_wcs_header).copy()

    # Normalize to IDL-GX conventions used in box.index.
    ctype1 = str(header.get("CTYPE1", ""))
    ctype2 = str(header.get("CTYPE2", ""))
    if ctype1.startswith("HGLN-"):
        header["CTYPE1"] = "CRLN-" + ctype1.split("-", 1)[1]
    if ctype2.startswith("HGLT-"):
        header["CTYPE2"] = "CRLT-" + ctype2.split("-", 1)[1]

    header["SIMPLE"] = int(bool(header.get("SIMPLE", 1)))
    header["BITPIX"] = int(header.get("BITPIX", 8))
    header["NAXIS"] = int(header.get("NAXIS", 2))

    if source_map is not None:
        date_obs = source_map.date.isot if source_map.date is not None else None
        if date_obs:
            header["DATE-OBS"] = date_obs
            header["DATE_OBS"] = date_obs
            header["DATE"] = date_obs
        elif "DATE-OBS" in header and "DATE_OBS" not in header:
            header["DATE_OBS"] = header["DATE-OBS"]

        if hasattr(source_map, "dsun") and source_map.dsun is not None:
            header["DSUN_OBS"] = float(source_map.dsun.to_value(u.m))

        obs = source_map.observer_coordinate
        if obs is not None:
            obs_hgs = obs.transform_to(HeliographicStonyhurst(obstime=source_map.date))
            header["HGLN_OBS"] = float(obs_hgs.lon.to_value(u.deg))
            header["HGLT_OBS"] = float(obs_hgs.lat.to_value(u.deg))
            header["SOLAR_B0"] = float(obs_hgs.lat.to_value(u.deg))
            try:
                obs_hgc = obs.transform_to(HeliographicCarrington(observer="earth", obstime=source_map.date))
                header["CRLN_OBS"] = float(obs_hgc.lon.to_value(u.deg))
                header["CRLT_OBS"] = float(obs_hgc.lat.to_value(u.deg))
            except Exception:
                # If Carrington transformation fails (e.g. unsupported observer configuration),
                # proceed without CRLN_OBS/CRLT_OBS rather than aborting header creation.
                # Carrington observer transforms can fail for some observer metadata;
                # keep header generation robust and proceed with available keys.
                pass

        if getattr(source_map, "rsun_meters", None) is not None:
            header["RSUN_REF"] = float(source_map.rsun_meters.to_value(u.m))

        tel = source_map.meta.get("telescop")
        if tel:
            header["TELESCOP"] = str(tel)
        instr = source_map.meta.get("instrume")
        if instr:
            header["INSTRUME"] = str(instr)
        if "WCSNAME" not in header:
            header["WCSNAME"] = "Carrington-Heliographic"

    return header.tostring(sep="\n", endcard=True)


def _print_info(cfg: Fov2BoxConfig) -> None:
    resolved_reduce_passed = cfg.reduce_passed if cfg.reduce_passed is not None else (0 if cfg.center_vox else 1)
    import pyampp
    import sunpy

    runtime_rows = [
        ("python_executable", sys.executable, "Interpreter used for this run"),
        ("python_version", sys.version.split()[0], "Python runtime version"),
        ("gx_fov2box_module", __file__, "Loaded gx_fov2box module path"),
        ("pyampp_module", pyampp.__file__, "Loaded pyampp package path"),
        ("sunpy_version", sunpy.__version__, "Loaded sunpy version"),
        ("gx-fov2box_cli", shutil.which("gx-fov2box"), "Resolved CLI entry point on PATH"),
    ]
    rows = [
        ("time", cfg.time, "Observation time in ISO format"),
        ("coords", cfg.coords, "Center coordinates: arcsec (HPC) or degrees (HGC/HGS)"),
        ("hpc/hgc/hgs", "HPC" if cfg.hpc else "HGC" if cfg.hgc else "HGS", "Coordinate frame"),
        ("cea/top", "TOP" if cfg.top else "CEA", "Basemap projection"),
        ("box_dims", cfg.box_dims, "Box dimensions (nx, ny, nz) in pixels"),
        ("dx_km", cfg.dx_km, "Voxel size in km"),
        ("pad_frac", cfg.pad_frac, "Padding fraction used for context map FOV"),
        ("data_dir", cfg.data_dir, "SDO download/cache directory"),
        ("gxmodel_dir", cfg.gxmodel_dir, "Output gx_models directory"),
        ("entry_box", cfg.entry_box, "Path to precomputed HDF5 box"),
        ("save_empty_box", cfg.save_empty_box, "Save NONE stage"),
        ("save_potential", cfg.save_potential, "Save POT stage"),
        ("save_bounds", cfg.save_bounds, "Save BND stage"),
        ("save_nas", cfg.save_nas, "Save NAS stage"),
        ("save_gen", cfg.save_gen, "Save NAS.GEN stage"),
        ("save_chr", cfg.save_chr, "Save NAS.CHR stage"),
        ("stop_after", cfg.stop_after, "Stop after stage (none/bnd/pot/nas/gen/chr)"),
        ("empty_box_only", cfg.empty_box_only, "Stop after NONE stage"),
        ("potential_only", cfg.potential_only, "Stop after POT stage"),
        ("nlfff_only", cfg.nlfff_only, "Stop after NAS stage"),
        ("generic_only", cfg.generic_only, "Stop after NAS.GEN stage"),
        ("use_potential", cfg.use_potential, "Skip NLFFF; reuse potential field"),
        ("center_vox", cfg.center_vox, "Compute lines only through voxel centers (sets reduce_passed=0 unless overridden)"),
        ("reduce_passed", cfg.reduce_passed, "Expert override: 0|1|2|3 (takes precedence over --center-vox)"),
        ("reduce_passed_resolved", resolved_reduce_passed, "Effective line-reduction mode used by tracer"),
        ("euv", cfg.euv, "Download AIA EUV context maps"),
        ("uv", cfg.uv, "Download AIA UV context maps"),
        ("sfq", cfg.sfq, "Use SFQ disambiguation (method=0)"),
        ("jump2potential", cfg.jump2potential, "Start from entry box and jump to POT"),
        ("jump2nlfff", cfg.jump2nlfff, "Start from entry box and jump to NAS"),
        ("jump2lines", cfg.jump2lines, "Start from entry box and jump to GEN"),
        ("jump2chromo", cfg.jump2chromo, "Start from entry box and jump to CHR"),
    ]
    print("gx-fov2box --info")
    for name, value, desc in runtime_rows + rows:
        print(f"- {name}: {value} :: {desc}")
    missing = []
    if not cfg.time:
        missing.append("--time")
    if not cfg.coords:
        missing.append("--coords")
    if not cfg.box_dims:
        missing.append("--box-dims")
    if missing:
        print("\nWarnings:")
        for item in missing:
            print(f"- Missing required {item}")
        if cfg.entry_box:
            print("- entry_box provided; time/box-dims may be inferred if present")


def _load_hmi_maps_from_downloader(
    time: Time,
    data_dir: Path,
    euv: bool,
    uv: bool,
    disambig_method: int = 2,
) -> tuple[Dict[str, Map], dict]:
    import time as time_mod

    downloader = SDOImageDownloader(time, data_dir=str(data_dir), euv=euv, uv=uv, hmi=True)
    missing_before = []
    if downloader.existence_report:
        for category in ("hmi_b", "hmi_m", "hmi_ic"):
            items = downloader.existence_report.get(category, {})
            missing_before.extend([k for k, exists in items.items() if not exists])
    t0 = time_mod.perf_counter()
    files = downloader.download_images()
    elapsed = time_mod.perf_counter() - t0
    downloaded = len(missing_before) > 0
    required = ["field", "inclination", "azimuth", "disambig", "continuum", "magnetogram"]
    missing = [k for k in required if not files.get(k)]
    if missing:
        raise RuntimeError(f"Missing required HMI files: {missing}")

    map_field = Map(files["field"])
    map_inclination = Map(files["inclination"])
    map_azimuth = Map(files["azimuth"])
    map_disambig = Map(files["disambig"])
    map_conti = Map(files["continuum"])
    map_losma = Map(files["magnetogram"])

    map_azimuth = hmi_disambig(map_azimuth, map_disambig, method=disambig_method)

    maps = {
        "field": map_field,
        "inclination": map_inclination,
        "azimuth": map_azimuth,
        "disambig": map_disambig,
        "continuum": map_conti,
        "magnetogram": map_losma,
    }
    for key, path in files.items():
        if key in ("field", "inclination", "azimuth", "disambig", "continuum", "magnetogram"):
            continue
        try:
            maps[f"AIA_{key}"] = Map(path)
        except Exception:
            # Skip optional context channels that cannot be loaded.
            continue
    info = {"downloaded": downloaded, "elapsed": elapsed, "missing_before": missing_before}
    return maps, info


def _make_header(map_field: Map) -> Dict[str, Any]:
    header_field = map_field.wcs.to_header()
    field_frame = map_field.center.heliographic_carrington.frame
    lon, lat = field_frame.lon.value, field_frame.lat.value
    obs_time = Time(map_field.date)
    dsun_obs = header_field["DSUN_OBS"]
    return {"lon": lon, "lat": lat, "dsun_obs": dsun_obs, "obs_time": str(obs_time.iso)}


def _make_gen_chromo(chromo_box: dict) -> dict:
    gen_keys = [
        "dr",
        "bcube",
        "start_idx",
        "end_idx",
        "av_field",
        "phys_length",
        "voxel_status",
        "apex_idx",
        "codes",
        "seed_idx",
    ]
    gen = {k: chromo_box[k] for k in gen_keys if k in chromo_box}
    if "attrs" in chromo_box:
        gen["attrs"] = chromo_box["attrs"]
    return gen


def _resolve_box_params(cfg: Fov2BoxConfig) -> Tuple[Time, Tuple[int, int, int], float]:
    obs_time = Time(cfg.time) if cfg.time else None
    box_dims = cfg.box_dims
    dx_km = cfg.dx_km
    if cfg.entry_box:
        entry_path = Path(cfg.entry_box)
        if entry_path.exists() and entry_path.suffix == ".h5":
            box_b3d = read_b3d_h5(str(entry_path))
            corona = box_b3d.get("corona")
            if corona is not None:
                if box_dims is None and "bx" in corona:
                    box_dims = corona["bx"].shape
                if "dr" in corona and dx_km == 0:
                    dr3 = corona["dr"]
                    dx_km = float(dr3[0] * sun_consts.radius.to(u.km).value)
        if obs_time is None:
            inferred = _infer_time_from_path(entry_path)
            if inferred:
                obs_time = Time(inferred)
    if obs_time is None:
        raise ValueError("--time is required unless it can be inferred from entry_box")
    if box_dims is None:
        raise ValueError("--box-dims is required unless it can be inferred from entry_box")
    return obs_time, tuple(int(v) for v in box_dims), float(dx_km)


@app.command()
def main(
    time: Optional[str] = typer.Option(None, "--time", help="Observation time ISO (e.g. 2024-05-12T00:00:00)"),
    coords: Optional[Tuple[float, float]] = typer.Option(None, "--coords", help="Center coords (x y)"),
    hpc: bool = typer.Option(False, "--hpc/--no-hpc", help="Use helioprojective coordinates"),
    hgc: bool = typer.Option(False, "--hgc/--no-hgc", help="Use heliographic carrington coordinates"),
    hgs: bool = typer.Option(False, "--hgs/--no-hgs", help="Use heliographic stonyhurst coordinates"),
    cea: bool = typer.Option(False, "--cea", help="Use CEA basemap projection"),
    top: bool = typer.Option(False, "--top", help="Use TOP-view basemap projection"),
    box_dims: Optional[Tuple[int, int, int]] = typer.Option(None, "--box-dims", help="Box dims in pixels"),
    dx_km: float = typer.Option(1400.0, "--dx-km", help="Voxel size in km"),
    pad_frac: float = typer.Option(0.10, "--pad-frac", help="Padding fraction for FOV"),
    data_dir: str = typer.Option(DOWNLOAD_DIR, "--data-dir", help="SDO data directory"),
    gxmodel_dir: str = typer.Option(GXMODEL_DIR, "--gxmodel-dir", help="GX model output directory"),
    entry_box: Optional[str] = typer.Option(None, "--entry-box", help="Existing HDF5 box"),
    save_empty_box: bool = typer.Option(False, "--save-empty-box", help="Save NONE stage"),
    save_potential: bool = typer.Option(False, "--save-potential", help="Save POT stage"),
    save_bounds: bool = typer.Option(False, "--save-bounds", help="Save BND stage"),
    save_nas: bool = typer.Option(False, "--save-nas", help="Save NAS stage"),
    save_gen: bool = typer.Option(False, "--save-gen", help="Save NAS.GEN stage"),
    save_chr: bool = typer.Option(False, "--save-chr", help="Save NAS.CHR stage"),
    stop_after: Optional[str] = typer.Option(None, "--stop-after", help="Stop after stage"),
    empty_box_only: bool = typer.Option(False, "--empty-box-only", help="Stop after NONE"),
    potential_only: bool = typer.Option(False, "--potential-only", help="Stop after POT"),
    nlfff_only: bool = typer.Option(False, "--nlfff-only", help="Stop after NAS"),
    generic_only: bool = typer.Option(False, "--generic-only", help="Stop after NAS.GEN"),
    use_potential: bool = typer.Option(False, "--use-potential", help="Skip NLFFF"),
    center_vox: bool = typer.Option(False, "--center-vox", help="Trace lines through voxel centers"),
    reduce_passed: Optional[int] = typer.Option(
        None,
        "--reduce-passed",
        min=0,
        max=3,
        help="Expert line tracing reduction bitmask: 0|1|2|3 (overrides --center-vox)",
    ),
    euv: bool = typer.Option(True, "--euv/--no-euv", help="Download AIA EUV maps"),
    uv: bool = typer.Option(True, "--uv/--no-uv", help="Download AIA UV maps"),
    sfq: bool = typer.Option(False, "--sfq", help="Use SFQ disambiguation (method=0)"),
    jump2potential: bool = typer.Option(False, "--jump2potential", help="Jump to POT"),
    jump2nlfff: bool = typer.Option(False, "--jump2nlfff", help="Jump to NAS"),
    jump2lines: bool = typer.Option(False, "--jump2lines", help="Jump to GEN"),
    jump2chromo: bool = typer.Option(False, "--jump2chromo", help="Jump to CHR"),
    info: bool = typer.Option(False, "--info", help="Show resolved defaults and exit"),
) -> None:
    cfg = Fov2BoxConfig(
        time=time,
        coords=coords,
        hpc=hpc,
        hgc=hgc,
        hgs=hgs,
        cea=cea,
        top=top,
        box_dims=box_dims,
        dx_km=dx_km,
        pad_frac=pad_frac,
        data_dir=data_dir,
        gxmodel_dir=gxmodel_dir,
        entry_box=entry_box,
        save_empty_box=save_empty_box,
        save_potential=save_potential,
        save_bounds=save_bounds,
        save_nas=save_nas,
        save_gen=save_gen,
        save_chr=save_chr,
        stop_after=stop_after,
        empty_box_only=empty_box_only,
        potential_only=potential_only,
        nlfff_only=nlfff_only,
        generic_only=generic_only,
        use_potential=use_potential,
        center_vox=center_vox,
        reduce_passed=reduce_passed,
        euv=euv,
        uv=uv,
        sfq=sfq,
        jump2potential=jump2potential,
        jump2nlfff=jump2nlfff,
        jump2lines=jump2lines,
        jump2chromo=jump2chromo,
        info=info,
    )

    if not any([cfg.hpc, cfg.hgc, cfg.hgs]):
        cfg.hpc = True
    if cfg.cea and cfg.top:
        raise ValueError("Select only one projection: --cea or --top")
    if not cfg.cea and not cfg.top:
        cfg.cea = True

    if cfg.empty_box_only:
        cfg.stop_after = cfg.stop_after or "none"
    elif cfg.potential_only:
        cfg.stop_after = cfg.stop_after or "pot"
    elif cfg.nlfff_only:
        cfg.stop_after = cfg.stop_after or "nas"
    elif cfg.generic_only:
        cfg.stop_after = cfg.stop_after or "gen"

    if cfg.info:
        _print_info(cfg)
        return

    import time as time_mod
    import warnings
    import logging

    t_start = time_mod.perf_counter()
    warnings.filterwarnings("ignore", message=".*assume_spherical_screen.*")
    logging.getLogger("reproject").setLevel(logging.WARNING)
    logging.getLogger("sunpy").setLevel(logging.WARNING)

    obs_time, box_dims_resolved, dx_km = _resolve_box_params(cfg)
    cfg.dx_km = dx_km

    if cfg.coords is None:
        raise ValueError("--coords is required")
    if sum([cfg.hpc, cfg.hgc, cfg.hgs]) != 1:
        raise ValueError("Select exactly one of --hpc, --hgc, --hgs")
    jump_flags_set = any([cfg.jump2potential, cfg.jump2nlfff, cfg.jump2lines, cfg.jump2chromo])
    if jump_flags_set and not cfg.entry_box:
        print("Warning: jump2* flags are ignored unless --entry-box is provided.")

    import time as time_mod

    observer = get_earth(obs_time)

    data_dir_path = Path(cfg.data_dir).expanduser().resolve()
    disambig_method = 0 if cfg.sfq else 2
    maps, download_info = _load_hmi_maps_from_downloader(
        obs_time, data_dir_path, cfg.euv, cfg.uv, disambig_method=disambig_method
    )
    if download_info["downloaded"]:
        print(f"HMI data: downloaded missing files in {download_info['elapsed']:.2f}s.")
    else:
        print("HMI data: already present in local repository.")

    rsun = u.Quantity(maps["field"].rsun_meters, u.m).to(u.km)

    if cfg.hpc:
        box_origin = SkyCoord(cfg.coords[0] * u.arcsec, cfg.coords[1] * u.arcsec,
                              obstime=obs_time, observer=observer, rsun=rsun, frame=Helioprojective)
    elif cfg.hgc:
        box_origin = SkyCoord(lon=cfg.coords[0] * u.deg, lat=cfg.coords[1] * u.deg,
                              radius=rsun, obstime=obs_time, observer=observer,
                              frame=HeliographicCarrington)
    else:
        box_origin = SkyCoord(lon=cfg.coords[0] * u.deg, lat=cfg.coords[1] * u.deg,
                              radius=rsun, obstime=obs_time, observer=observer,
                              frame=HeliographicStonyhurst)

    box_dims_q = u.Quantity(list(box_dims_resolved)) * u.pix
    box_res = (cfg.dx_km * u.km).to(u.Mm)

    frame_obs = Helioprojective(observer=observer, obstime=obs_time, rsun=rsun)
    frame_hcc = Heliocentric(observer=box_origin, obstime=obs_time)
    box_center = box_origin.transform_to(frame_hcc)
    center_z = box_center.z + (box_dims_q[2] / u.pix * box_res) / 2
    box_center = SkyCoord(x=box_center.x, y=box_center.y, z=center_z, frame=frame_hcc)

    box = Box(frame_obs, box_origin, box_center, box_dims_q, box_res)
    if cfg.top:
        bottom_wcs_header = box.bottom_top_header(dsun_obs=maps["field"].dsun)
        projection_tag = "TOP"
    else:
        bottom_wcs_header = box.bottom_cea_header
        projection_tag = "CEA"
    fov_coords = box.bounds_coords_bl_tr(pad_frac=cfg.pad_frac)

    map_bp, map_bt, map_br = hmi_b2ptr(maps["field"], maps["inclination"], maps["azimuth"])

    def submap_with_fov(_map: Map) -> Map:
        bl = fov_coords[0].transform_to(_map.coordinate_frame)
        tr = fov_coords[1].transform_to(_map.coordinate_frame)
        return _map.submap(bl, top_right=tr)

    map_bp = submap_with_fov(map_bp)
    map_bt = submap_with_fov(map_bt)
    map_br = submap_with_fov(map_br)
    map_cont = submap_with_fov(maps["continuum"])
    map_los = submap_with_fov(maps["magnetogram"])

    map_bx = Map(-map_bt.data, map_bt.meta)
    map_by = map_bp
    map_bz = map_br

    bottom_bx = map_bx.reproject_to(bottom_wcs_header, algorithm="exact")
    bottom_by = map_by.reproject_to(bottom_wcs_header, algorithm="exact")
    bottom_bz = map_bz.reproject_to(bottom_wcs_header, algorithm="exact")

    base_bz = map_los.reproject_to(bottom_wcs_header, algorithm="exact")
    base_ic = map_cont.reproject_to(bottom_wcs_header, algorithm="exact")

    index = _build_index_header(bottom_wcs_header, bottom_bz)
    chromo_mask = decompose(base_bz.data.T, base_ic.data.T)
    base_group = {
        "bx": bottom_bx.data,
        "by": bottom_by.data,
        "bz": bottom_bz.data,
        "ic": base_ic.data,
        "chromo_mask": chromo_mask,
        "index": index,
    }

    refmaps = {}
    def add_refmap(ref_id: str, smap: Map) -> None:
        smap = submap_with_fov(smap)
        # Use WCS header only to avoid non-ASCII FITS meta from some sources.
        header = smap.wcs.to_header()
        refmaps[ref_id] = {"data": smap.data, "wcs_header": header.tostring(sep="\\n", endcard=True)}

    add_refmap("Bz_reference", maps["magnetogram"])
    add_refmap("Ic_reference", maps["continuum"])

    vert_current_error = None
    try:
        vc_header = map_bx.wcs.to_header().tostring(sep="\\n", endcard=True)
        rsun_arcsec = maps["magnetogram"].rsun_obs.to_value(u.arcsec)
        crpix1, crpix2 = map_bx.wcs.wcs.crpix
        cdelt1 = map_bx.scale.axis1.to_value(u.arcsec / u.pix)
        cdelt2 = map_bx.scale.axis2.to_value(u.arcsec / u.pix)
        jz = compute_vertical_current(map_bz.data, map_bx.data, map_by.data,
                                      vc_header, rsun_arcsec,
                                      crpix1=crpix1, crpix2=crpix2,
                                      cdelt1_arcsec=cdelt1, cdelt2_arcsec=cdelt2)
        refmaps["Vert_current"] = {"data": jz, "wcs_header": vc_header}
    except Exception as exc:
        vert_current_error = str(exc)

    for pb in AIA_EUV_PASSBANDS + AIA_UV_PASSBANDS:
        key = f"AIA_{pb}"
        if key in maps:
            add_refmap(key, maps[key])

    obs_dr = (cfg.dx_km * u.km) / rsun
    dr3 = np.array([obs_dr.value, obs_dr.value, obs_dr.value])
    empty_grid = np.zeros((box_dims_resolved[0], box_dims_resolved[1], box_dims_resolved[2]), dtype=float)
    default_grid = {
        "voxel_id": gx_box2id({"corona": {"bx": empty_grid, "by": empty_grid, "bz": empty_grid, "dr": dr3}}),
        "dx": float(dr3[0]),
        "dy": float(dr3[1]),
        # Keep compact for uniform-grid models: dz = corona/dr[2].
        "dz": np.array([float(dr3[2])], dtype=float),
    }

    out_dir = _stage_output_dir(cfg.gxmodel_dir, obs_time)
    coord_tag = _format_coord_tag(box_origin.transform_to(HeliographicCarrington(obstime=obs_time)).lon.to_value(u.deg),
                                  box_origin.transform_to(HeliographicCarrington(obstime=obs_time)).lat.to_value(u.deg))
    base = _stage_file_base(obs_time, coord_tag, projection_tag=projection_tag)
    execute_cmd = _build_execute_cmd(cfg)

    produced = []
    stage_times = {}

    def finalize() -> None:
        if produced:
            print("\nCompleted gx-fov2box. Output files:")
            for path in produced:
                print(f"- {path}")
        if stage_times:
            print("Stage timings:")
            for stage, seconds in stage_times.items():
                print(f"- {stage}: {seconds:.2f}s")
        total = time_mod.perf_counter() - t_start
        print(f"Total elapsed: {total:.2f}s")

    def save_stage(stage_tag: str, stage_box: dict) -> None:
        if not _should_save_stage(stage_tag, cfg):
            return
        stage_box = dict(stage_box)
        if "base" not in stage_box:
            stage_box["base"] = base_group
        if "refmaps" not in stage_box:
            stage_box["refmaps"] = refmaps
        if "grid" not in stage_box:
            stage_box["grid"] = default_grid
        stage_id = f"{base}.{stage_tag}"
        metadata = {
            "execute": execute_cmd,
            "id": stage_id,
            "disambiguation": "SFQ" if cfg.sfq else "HMI",
            "projection": projection_tag,
        }
        if vert_current_error:
            metadata["vert_current_error"] = vert_current_error
        stage_box["metadata"] = metadata
        out_path = _stage_filename(out_dir, base, stage_tag)
        write_b3d_h5(str(out_path), stage_box)
        produced.append(out_path)

    if cfg.save_empty_box or cfg.empty_box_only or cfg.stop_after in ("none", "empty", "empty_box"):
        t0 = time_mod.perf_counter()
        empty = np.zeros((box_dims_resolved[0], box_dims_resolved[1], box_dims_resolved[2]), dtype=float)
        stage_box = {"corona": {"bx": empty, "by": empty, "bz": empty, "dr": dr3, "attrs": {"model_type": "none"}}}
        save_stage("NONE", stage_box)
        stage_times["NONE"] = time_mod.perf_counter() - t0
        if cfg.empty_box_only or _last_stage_tag(cfg.stop_after) == "NONE":
            finalize()
            return

    if cfg.save_bounds or cfg.stop_after in ("bnd", "bounds"):
        t0 = time_mod.perf_counter()
        stage_box = {"bounds": {"bx": bottom_bx.data, "by": bottom_by.data, "bz": bottom_bz.data, "dr": dr3}}
        save_stage("BND", stage_box)
        stage_times["BND"] = time_mod.perf_counter() - t0
        if _last_stage_tag(cfg.stop_after) == "BND":
            finalize()
            return

    active_jump = None
    if cfg.entry_box:
        if cfg.jump2potential:
            active_jump = "potential"
        elif cfg.jump2nlfff:
            active_jump = "nlfff"
        elif cfg.jump2lines:
            active_jump = "lines"
        elif cfg.jump2chromo:
            active_jump = "chromo"

    entry_corona = None
    entry_model = None
    entry_chromo = None
    if active_jump:
        loaded = read_b3d_h5(cfg.entry_box)
        entry_corona = loaded.get("corona")
        if entry_corona:
            entry_model = entry_corona.get("attrs", {}).get("model_type")
        entry_chromo = loaded.get("chromo")

    goto_lines = active_jump in ("lines", "chromo")
    goto_chromo = active_jump == "chromo"

    if goto_lines:
        if not entry_corona or entry_model not in ("pot", "nlfff"):
            raise ValueError("--jump2lines/--jump2chromo requires --entry-box with corona model_type=pot|nlfff")
        nlfff_box = entry_corona

    if not goto_lines:
        if active_jump == "nlfff":
            if not entry_corona or entry_model not in ("pot", "nlfff"):
                raise ValueError("--jump2nlfff requires --entry-box with corona model_type=pot|nlfff")
            pot_box = entry_corona
        else:
            t0 = time_mod.perf_counter()
            bnddata = bottom_bz.data.copy()
            bnddata[np.isnan(bnddata)] = 0.0
            maglib_lff = MagFieldLinFFF()
            maglib_lff.set_field(bnddata)
            pot_res = maglib_lff.LFFF_cube(nz=box_dims_resolved[2], alpha=0.0)
            pot_box = {
                "bx": pot_res["by"].swapaxes(0, 1),
                "by": pot_res["bx"].swapaxes(0, 1),
                "bz": pot_res["bz"].swapaxes(0, 1),
                "dr": dr3,
                "attrs": {"model_type": "pot"},
            }
            save_stage("POT", {"corona": pot_box})
            stage_times["POT"] = time_mod.perf_counter() - t0
            if cfg.potential_only or _last_stage_tag(cfg.stop_after) == "POT":
                finalize()
                return
        if cfg.use_potential:
            nlfff_box = {
                "bx": pot_box["bx"],
                "by": pot_box["by"],
                "bz": pot_box["bz"],
                "dr": dr3,
                "attrs": {"model_type": "pot"},
            }
        else:
            t0 = time_mod.perf_counter()
            maglib = MagFieldProcessor()
            pot_res = {
                "bx": pot_box["by"].swapaxes(0, 1),
                "by": pot_box["bx"].swapaxes(0, 1),
                "bz": pot_box["bz"].swapaxes(0, 1),
            }
            maglib.load_cube_vars(pot_res)
            res_nlf = maglib.NLFFF()
            nlfff_box = {
                "bx": res_nlf["by"].swapaxes(0, 1),
                "by": res_nlf["bx"].swapaxes(0, 1),
                "bz": res_nlf["bz"].swapaxes(0, 1),
                "dr": dr3,
                "attrs": {"model_type": "nlfff"},
            }
            save_stage("NAS", {"corona": nlfff_box})
            stage_times["NAS"] = time_mod.perf_counter() - t0
            if cfg.nlfff_only or _last_stage_tag(cfg.stop_after) == "NAS":
                finalize()
                return

    if goto_chromo:
        required_line_keys = [
            "codes", "apex_idx", "start_idx", "end_idx", "seed_idx",
            "av_field", "phys_length", "voxel_status",
        ]
        if not entry_chromo or any(k not in entry_chromo for k in required_line_keys):
            raise ValueError("--jump2chromo requires --entry-box containing chromo line metadata (GEN stage).")
        lines = entry_chromo
        compute_lines_time = 0.0
    else:
        t0 = time_mod.perf_counter()
        maglib = MagFieldProcessor()
        maglib.load_cube_vars({
            "bx": nlfff_box["by"].swapaxes(0, 1),
            "by": nlfff_box["bx"].swapaxes(0, 1),
            "bz": nlfff_box["bz"].swapaxes(0, 1),
        })
        resolved_reduce_passed = cfg.reduce_passed if cfg.reduce_passed is not None else (0 if cfg.center_vox else 1)
        lines = maglib.lines(seeds=None, reduce_passed=resolved_reduce_passed)
        compute_lines_time = time_mod.perf_counter() - t0

    # base_bz/base_ic are prepared above for the base group

    header = _make_header(maps["field"])
    chromo_box = combo_model(nlfff_box, dr3, base_bz.data.T, base_ic.data.T)
    for k in ["codes", "apex_idx", "start_idx", "end_idx", "seed_idx",
              "av_field", "phys_length", "voxel_status"]:
        chromo_box[k] = lines[k]
    chromo_box["phys_length"] *= dr3[0]
    chromo_box["attrs"] = header

    if not goto_chromo:
        gen_chromo = _make_gen_chromo(chromo_box)
        save_stage("NAS.GEN", {"chromo": gen_chromo})
        stage_times["NAS.GEN"] = compute_lines_time
        if cfg.generic_only or _last_stage_tag(cfg.stop_after) == "NAS.GEN":
            finalize()
            return

    t0 = time_mod.perf_counter()
    chr_grid = {
        "voxel_id": gx_box2id({"corona": nlfff_box, "chromo": chromo_box}),
        "dx": float(dr3[0]),
        "dy": float(dr3[1]),
        "dz": chromo_box["dz"] if "dz" in chromo_box else dr3,
    }
    save_stage("NAS.CHR", {"chromo": chromo_box, "grid": chr_grid})
    stage_times["NAS.CHR"] = time_mod.perf_counter() - t0

    finalize()


if __name__ == "__main__":
    app()
