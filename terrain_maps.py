"""
DEM, slope and aspect maps for a village/AOI boundary KML, drawn over a Google
satellite basemap.

Usage:
    export OPENTOPOGRAPHY_API_KEY="your_key_here"        # Linux / macOS
    setx OPENTOPOGRAPHY_API_KEY "your_key_here"          # Windows (reopen terminal)

    python terrain_maps.py --kml "/path/to/Village Boundary.kml" --spring-name "Dhara 1"
    python test.py --kml boundary.kml --no-basemap
    python test.py --kml boundary.kml --interval 10         # 10 m contours on the slope map
    python test.py --kml boundary.kml --zoom 17 --buffer 300 --output-dir ./out
"""

import argparse
import os
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path

import fiona
import geopandas as gpd
import matplotlib.pyplot as plt
import mercantile
import numpy as np
import rasterio
import requests
from matplotlib.ticker import ScalarFormatter
from osgeo import gdal
from PIL import Image
from rasterio.enums import Resampling
from rasterio.mask import mask
from rasterio.transform import array_bounds, from_bounds
from rasterio.warp import calculate_default_transform, reproject, transform_bounds

warnings.filterwarnings("ignore")
fiona.drvsupport.supported_drivers["KML"] = "rw"
gdal.UseExceptions()

API_KEY_ENV = "OPENTOPOGRAPHY_API_KEY"
GOOGLE_SAT_URL = "https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}"
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"


# ---------------------------------------------------------------------------
# DEM download / processing
# ---------------------------------------------------------------------------
def download_dem(bounds, api_key, dem_type="SRTMGL1", output="dem.tif", log=print):
    west, south, east, north = bounds
    url = "https://portal.opentopography.org/API/globaldem"
    params = {
        "demtype": dem_type, "south": south, "north": north,
        "west": west, "east": east, "outputFormat": "GTiff", "API_Key": api_key,
    }

    response = requests.get(url, params=params, timeout=(15, 120))

    # Validation: check that the response is actually a TIFF file
    if response.status_code == 200 and response.content[:4] in [b"II\x2a\x00", b"MM\x00\x2a"]:
        with open(output, "wb") as f:
            f.write(response.content)
        log(f"✅ Downloaded {dem_type} to {Path(output).name}")
        return output

    # Never include the request URL in messages (it contains the API key)
    if response.status_code in (401, 403):
        reason = "OpenTopography rejected the API key (check it is correct and active)."
    elif response.status_code == 429:
        reason = "OpenTopography rate limit reached. Wait a while and try again."
    elif len(response.text) < 200:
        reason = response.text.strip()
    else:
        reason = "HTML error page (rate limit or auth issue)"
    raise RuntimeError(f"DEM download failed (HTTP {response.status_code}): {reason}")


def upscale_dem(input_path, subdivide_factor=5, output_path="upscaled_dem.tif", log=print):
    """Memory-efficient upscaling using Rasterio's internal resampling."""
    log(f"🚀 Resampling DEM by factor {subdivide_factor}...")
    with rasterio.open(input_path) as src:
        new_height = int(src.height * subdivide_factor)
        new_width = int(src.width * subdivide_factor)

        new_transform = src.transform * src.transform.scale(
            (src.width / new_width),
            (src.height / new_height),
        )

        profile = src.profile.copy()
        profile.update({
            "transform": new_transform,
            "height": new_height,
            "width": new_width,
            "nodata": src.nodata if src.nodata is not None else -9999,
        })

        data = src.read(
            out_shape=(src.count, new_height, new_width),
            resampling=Resampling.bilinear,
        )

        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(data)

    log("✅ Upscale complete")
    return output_path


def calculate_slope_aspect(dem_file, gdf, slope_file="slope.tif", aspect_file="aspect.tif"):
    avg_lat = gdf.geometry.centroid.y.mean()
    dynamic_scale = 111320 * np.cos(np.radians(avg_lat))
    gdal.DEMProcessing(str(slope_file), str(dem_file), "slope",
                       slopeFormat="percent", scale=dynamic_scale, computeEdges=True)
    gdal.DEMProcessing(str(aspect_file), str(dem_file), "aspect", computeEdges=True)
    return slope_file, aspect_file


# ---------------------------------------------------------------------------
# Google satellite basemap
# ---------------------------------------------------------------------------
def _fetch_tile(tile):
    """Download one tile; returns (tile, RGBA image) or (tile, None) on failure."""
    url = GOOGLE_SAT_URL.format(x=tile.x, y=tile.y, z=tile.z)
    try:
        r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=10)
        r.raise_for_status()
        return tile, Image.open(BytesIO(r.content)).convert("RGBA")
    except Exception:
        return tile, None


def download_google_basemap(bounds_wgs84, zoom=18, max_tiles=1200, workers=8, log=print):
    """Download and stitch Google satellite tiles covering bounds (west, south, east, north).

    Returns (rgba_array[H, W, 4], transform, crs) in EPSG:3857, or None on failure.
    The zoom is lowered automatically if the area would need more than max_tiles tiles.
    """
    west, south, east, north = bounds_wgs84
    tiles = list(mercantile.tiles(west, south, east, north, zoom))
    while len(tiles) > max_tiles and zoom > 10:
        zoom -= 1
        tiles = list(mercantile.tiles(west, south, east, north, zoom))
    log(f"🛰️ Downloading {len(tiles)} Google satellite tiles at zoom {zoom}...")

    min_x, max_x = min(t.x for t in tiles), max(t.x for t in tiles)
    min_y, max_y = min(t.y for t in tiles), max(t.y for t in tiles)
    size = 256
    canvas_w = (max_x - min_x + 1) * size
    canvas_h = (max_y - min_y + 1) * size
    canvas = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))

    n_ok = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for tile, img in pool.map(_fetch_tile, tiles):
            if img is not None:
                canvas.paste(img, ((tile.x - min_x) * size, (tile.y - min_y) * size))
                n_ok += 1
    if n_ok == 0:
        log("⚠️ No basemap tiles could be downloaded.")
        return None
    if n_ok < len(tiles):
        log(f"⚠️ {len(tiles) - n_ok}/{len(tiles)} tiles failed and were left transparent.")

    ul = mercantile.xy_bounds(min_x, min_y, zoom)
    lr = mercantile.xy_bounds(max_x, max_y, zoom)
    transform = from_bounds(ul.left, lr.bottom, lr.right, ul.top, canvas_w, canvas_h)
    log("✅ Basemap ready.")
    return np.array(canvas), transform, "EPSG:3857"


def basemap_to_crs(basemap, target_crs):
    """Reproject the stitched mosaic to target_crs. Returns (RGBA image, [left, right, bottom, top])."""
    rgba, src_transform, src_crs = basemap
    h, w = rgba.shape[:2]
    src_bounds = array_bounds(h, w, src_transform)  # west, south, east, north
    dst_transform, dw, dh = calculate_default_transform(src_crs, target_crs, w, h, *src_bounds)

    out = np.zeros((dh, dw, 4), dtype=np.uint8)
    for i in range(4):
        band = np.zeros((dh, dw), dtype=np.uint8)
        reproject(
            source=np.ascontiguousarray(rgba[:, :, i]), destination=band,
            src_transform=src_transform, src_crs=src_crs,
            dst_transform=dst_transform, dst_crs=target_crs,
            resampling=Resampling.bilinear,
        )
        out[:, :, i] = band
    left, bottom, right, top = array_bounds(dh, dw, dst_transform)
    return out, [left, right, bottom, top]


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def _read_masked(src):
    data = src.read(1).astype("float64")
    if src.nodata is not None:
        data = np.ma.masked_equal(data, src.nodata)
    return np.ma.masked_invalid(data)


def create_visualizations(dem_path, slope_path, aspect_path, gdf,
                          output_dir=Path("."), basemap=None, show=False, interval=5.0,
                          spring_name="", log=print):
    """Plots with Lat/Long, colorbars, dynamic scale bar, optional Google satellite basemap,
    and elevation contours (every `interval` metres) on the slope map."""
    log(f"🖼️ Generating visualization plots (slope map with {interval:g} m contours)...")
    suffix = f" - {spring_name}" if spring_name else ""
    output_dir = Path(output_dir)
    layers = [
        (dem_path, "Elevation (m)", "terrain", "dem_plot.png", f"DEM{suffix}", None, None, False),
        (slope_path, "Slope (%)", "RdYlGn_r", "slope_plot.png",
         f"Slope Analysis with {interval:g} m Contours{suffix}", 0, 100, True),
        (aspect_path, "Aspect (°)", "twilight", "aspect_plot.png", f"Aspect Analysis{suffix}", 0, 360, False),
    ]

    # Elevation grid + contour levels from the cropped DEM
    with rasterio.open(dem_path) as dem_src:
        elev = _read_masked(dem_src)
    n_levels = (float(elev.max()) - float(elev.min())) / interval
    if n_levels > 400:
        raise ValueError(
            f"Contour interval {interval:g} m would draw about {int(n_levels)} lines over "
            f"{float(elev.max()) - float(elev.min()):.0f} m of relief. Use a larger interval.")
    levels = np.arange(np.floor(elev.min() / interval) * interval,
                       elev.max() + interval, interval)
    contour_grid = np.flipud(elev)  # contour() draws with origin at the bottom

    # Reproject the basemap once; all layers share the same CRS
    base_img = base_extent = None
    if basemap is not None:
        with rasterio.open(dem_path) as ref:
            base_img, base_extent = basemap_to_crs(basemap, ref.crs)

    for path, cbar_label, cmap, save_name, title, vmin, vmax, draw_contours in layers:
        with rasterio.open(path) as src:
            data = _read_masked(src)
            extent = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]

            fig, ax = plt.subplots(figsize=(10, 8))

            if base_img is not None:
                ax.imshow(base_img, extent=base_extent, zorder=0)
            alpha = 0.6 if base_img is not None else 1.0
            im = ax.imshow(data, cmap=cmap, extent=extent, vmin=vmin, vmax=vmax,
                           alpha=alpha, zorder=1)

            cbar = fig.colorbar(im, ax=ax, shrink=0.7)
            cbar.set_label(cbar_label)

            ax.set_title(title, fontweight="bold")
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            for axis in (ax.xaxis, ax.yaxis):
                fmt = ScalarFormatter(useOffset=False)
                fmt.set_scientific(False)
                axis.set_major_formatter(fmt)

            if draw_contours:
                cs = ax.contour(contour_grid, levels=levels, extent=extent, colors="black",
                                linewidths=0.5, alpha=0.8, zorder=2)
                ax.clabel(cs, inline=True, fontsize=7, fmt="%1.0f m")

            gdf.plot(ax=ax, facecolor="none", edgecolor="black", linewidth=1.5, zorder=3)

            # Keep the view on the DEM area (the basemap mosaic is slightly larger)
            ax.set_xlim(extent[0], extent[1])
            ax.set_ylim(extent[2], extent[3])

            # 1. North direction indicator
            ax.text(0.95, 0.95, "N", transform=ax.transAxes, ha="center", weight="bold", fontsize=12)
            ax.annotate("", xy=(0.95, 0.93), xytext=(0.95, 0.85),
                        arrowprops=dict(arrowstyle="->"), xycoords="axes fraction",
                        textcoords="axes fraction")

            # 2. Map source
            source = "Source: OpenTopography"
            if base_img is not None:
                source += " | Basemap: Google Satellite"
            ax.text(0.98, 0.01, source, transform=ax.transAxes, ha="right", fontsize=8,
                    bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", pad=1.5))

            # 3. Automatic dynamic scale bar
            x_min, x_max = ax.get_xlim()
            y_min, y_max = ax.get_ylim()
            avg_lat = (y_min + y_max) / 2
            width_m = (x_max - x_min) * (111320 * np.cos(np.radians(avg_lat)))

            if width_m > 20000: scale_len, scale_lbl = 10000, "10 km"
            elif width_m > 5000: scale_len, scale_lbl = 2000, "2 km"
            elif width_m > 2000: scale_len, scale_lbl = 1000, "1 km"
            elif width_m > 1000: scale_len, scale_lbl = 500, "500 m"
            else: scale_len, scale_lbl = 100, "100 m"

            scale_ratio = scale_len / width_m
            ax.plot([0.05, 0.05 + scale_ratio], [0.05, 0.05], transform=ax.transAxes,
                    color="black", linewidth=3)
            ax.text(0.05 + (scale_ratio / 2), 0.07, scale_lbl, transform=ax.transAxes,
                    ha="center", va="bottom", fontsize=9, fontweight="bold",
                    bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", pad=1.5))

            out_file = output_dir / save_name
            plt.savefig(out_file, dpi=300, bbox_inches="tight")
            if show:
                plt.show()
            plt.close(fig)
            log(f"Saved {out_file.name}")

    return "Plots Saved"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(
        description="Download a DEM for a KML boundary and generate DEM, slope and aspect maps "
                    "over a Google satellite basemap.")
    parser.add_argument("--kml", required=True, type=Path, help="Input boundary KML file")
    parser.add_argument("--output-dir", type=Path, default=Path("."),
                        help="Folder for all outputs (default: current directory)")
    parser.add_argument("--interval", type=float, default=5.0,
                        help="Contour interval in metres on the slope map (default 5)")
    parser.add_argument("--buffer", type=float, default=200,
                        help="Buffer in metres around the boundary for the DEM download (default 200)")
    parser.add_argument("--dem-type", default="SRTMGL1",
                        help="OpenTopography DEM type, e.g. SRTMGL1, COP30 (default SRTMGL1)")
    parser.add_argument("--subdivide", type=int, default=5,
                        help="DEM upscale factor (default 5)")
    parser.add_argument("--zoom", type=int, default=18,
                        help="Google satellite zoom level (default 18; auto-lowered for large areas)")
    parser.add_argument("--spring-name", default="", help="Spring/site name added to map titles")
    parser.add_argument("--no-basemap", action="store_true",
                        help="Skip the Google satellite basemap")
    parser.add_argument("--show", action="store_true", help="Also open each plot in a window")
    args = parser.parse_args(argv)

    args.kml = args.kml.expanduser().resolve()
    if not args.kml.is_file():
        parser.error(f"KML file does not exist: {args.kml}")
    if args.kml.suffix.lower() != ".kml":
        parser.error("--kml must point to a .kml file")
    if args.interval <= 0:
        parser.error("--interval must be greater than 0")
    args.output_dir = args.output_dir.expanduser().resolve()
    return args


MAX_EXTENT_DEG = 0.3  # ~33 km per side; guards against runaway memory use


def run_pipeline(kml, output_dir, api_key, spring_name="", interval=5.0, buffer=200.0,
                 dem_type="SRTMGL1", subdivide=5, zoom=18, use_basemap=True,
                 show=False, log=print, progress=None):
    """Full workflow. Returns a dict of output file paths. Raises on failure."""
    progress = progress or (lambda pct: None)
    if not api_key:
        raise ValueError("An OpenTopography API key is required.")
    if interval <= 0:
        raise ValueError("Contour interval must be greater than 0.")
    if buffer < 0:
        raise ValueError("Buffer cannot be negative.")
    if subdivide < 1:
        raise ValueError("Upscale factor must be at least 1.")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    progress(3)
    gdf = gpd.read_file(kml, driver="KML")
    if gdf.empty:
        raise ValueError("The KML contains no features.")
    if not gdf.geom_type.isin(["Polygon", "MultiPolygon"]).any():
        raise ValueError("The KML must contain a polygon boundary (found only points/lines).")
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    gdf = gdf.to_crs("EPSG:4326")

    # Download with buffer
    gdf_buffered = gdf.to_crs(gdf.estimate_utm_crs()).buffer(buffer).to_crs("EPSG:4326")
    w, s, e, n = gdf_buffered.total_bounds
    if (e - w) > MAX_EXTENT_DEG or (n - s) > MAX_EXTENT_DEG:
        raise ValueError(f"Area is too large (limit about {MAX_EXTENT_DEG}° ≈ 33 km per side "
                         f"including buffer). Use a smaller boundary or buffer.")
    log("📍 Boundary loaded.")

    progress(8)
    raw_dem = download_dem(gdf_buffered.total_bounds, api_key, dem_type=dem_type,
                           output=str(out / "dem.tif"), log=log)

    # 1. Upscale the whole tile subset first
    progress(25)
    high_res_dem = upscale_dem(raw_dem, subdivide_factor=subdivide,
                               output_path=str(out / "upscaled_dem.tif"), log=log)

    # 2. Mask/clip to the exact KML boundary
    progress(40)
    cropped_dem = out / "cropped_dem.tif"
    with rasterio.open(high_res_dem) as src:
        out_img, out_trans = mask(src, gdf.geometry, crop=True)
        out_meta = src.meta.copy()
        out_meta.update({
            "height": out_img.shape[1],
            "width": out_img.shape[2],
            "transform": out_trans,
        })
        with rasterio.open(cropped_dem, "w", **out_meta) as dest:
            dest.write(out_img)
        crop_bounds = transform_bounds(src.crs, "EPSG:4326", *rasterio.transform.array_bounds(
            out_img.shape[1], out_img.shape[2], out_trans))
    log("✂️ DEM clipped to boundary.")

    # 3. Google satellite basemap covering the cropped area (+2% margin)
    progress(50)
    basemap = None
    if use_basemap:
        bw, bs, be, bn = crop_bounds
        pad_x, pad_y = (be - bw) * 0.02, (bn - bs) * 0.02
        try:
            basemap = download_google_basemap((bw - pad_x, bs - pad_y, be + pad_x, bn + pad_y),
                                              zoom=zoom, log=log)
        except Exception as exc:
            log(f"⚠️ Basemap failed ({exc}); continuing without it.")

    # 4. Terrain products + plots
    progress(72)
    slope, aspect = calculate_slope_aspect(cropped_dem, gdf,
                                           slope_file=out / "slope.tif",
                                           aspect_file=out / "aspect.tif")
    progress(78)
    create_visualizations(cropped_dem, slope, aspect, gdf,
                          output_dir=out, basemap=basemap, show=show,
                          interval=interval, spring_name=spring_name, log=log)
    progress(100)
    log(f"Done. Outputs in {out}")
    return {
        "maps": [out / "dem_plot.png", out / "slope_plot.png", out / "aspect_plot.png"],
        "rasters": [cropped_dem, Path(slope), Path(aspect)],
    }


def main(argv=None):
    args = parse_arguments(argv)
    api_key = os.environ.get(API_KEY_ENV, "").strip()
    if not api_key:
        sys.exit(f"❌ Environment variable {API_KEY_ENV} is not set. "
                 f"Set it first, e.g.  export {API_KEY_ENV}=your_key")
    try:
        run_pipeline(args.kml, args.output_dir, api_key, spring_name=args.spring_name,
                     interval=args.interval, buffer=args.buffer, dem_type=args.dem_type,
                     subdivide=args.subdivide, zoom=args.zoom,
                     use_basemap=not args.no_basemap, show=args.show)
    except Exception as exc:
        sys.exit(f"❌ {str(exc).replace(api_key, '***')}")


if __name__ == "__main__":
    main()
