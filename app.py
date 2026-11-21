"""
Web front-end for terrain_maps.py.

Run:
    pip install -r requirements.txt      (GDAL must be installed separately, see README)
    python app.py                        -> http://127.0.0.1:8080

Environment variables:
    HOST   bind address (default 127.0.0.1; use 0.0.0.0 to expose on your network)
    PORT   port (default 8080)

Design notes
- Jobs run in a background thread, ONE AT A TIME (matplotlib/GDAL are heavy and not
  thread-safe in the way this script uses them). Others show as "queued".
- The OpenTopography key is used only inside the worker thread. It is never stored
  in the job record, never written to disk, and is scrubbed from any error text.
- Jobs and their files are kept in memory/temp for 6 hours, then deleted.
"""

import matplotlib

matplotlib.use("Agg")  # must happen before anything imports pyplot

import os
import re
import shutil
import tempfile
import threading
import time
import uuid
import zipfile
from pathlib import Path

from flask import Flask, abort, jsonify, render_template, request, send_from_directory

import terrain_maps

JOB_ROOT = Path(tempfile.gettempdir()) / "terrain_web_jobs"
JOB_ROOT.mkdir(parents=True, exist_ok=True)
JOB_TTL_SECONDS = 6 * 3600
DEM_TYPES = {"SRTMGL1", "SRTMGL3", "NASADEM", "COP30", "COP90", "AW3D30"}
DOWNLOADABLE = {
    "dem_plot.png", "slope_plot.png", "aspect_plot.png",
    "cropped_dem.tif", "slope.tif", "aspect.tif", "terrain_maps.zip",
}
JOB_ID_RE = re.compile(r"^[0-9a-f]{32}$")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024  # KML upload limit

jobs = {}
jobs_lock = threading.Lock()
run_lock = threading.Lock()  # serialises the heavy work


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _number(form, name, default, lo, hi, integer=False):
    raw = (form.get(name) or "").strip()
    if raw == "":
        return default
    try:
        value = int(raw) if integer else float(raw)
    except ValueError:
        raise ValueError(f"'{name}' must be a number.")
    if not (lo <= value <= hi):
        raise ValueError(f"'{name}' must be between {lo:g} and {hi:g}.")
    return value


def _cleanup_old_jobs():
    now = time.time()
    with jobs_lock:
        stale = [jid for jid, j in jobs.items()
                 if now - j["created"] > JOB_TTL_SECONDS and j["status"] in ("done", "error")]
        for jid in stale:
            jobs.pop(jid, None)
    for folder in JOB_ROOT.iterdir():
        try:
            if folder.is_dir() and now - folder.stat().st_mtime > JOB_TTL_SECONDS:
                shutil.rmtree(folder, ignore_errors=True)
        except OSError:
            pass


def _slug(text):
    return re.sub(r"[^A-Za-z0-9_-]+", "_", text).strip("_") or "terrain_maps"


def _worker(job_id, folder, kml_path, api_key, params):
    job = jobs[job_id]

    def log(msg):
        text = str(msg).replace(api_key, "***")
        with jobs_lock:
            job["log"].append(text)

    def progress(pct):
        job["progress"] = pct

    with run_lock:
        job["status"] = "running"
        try:
            result = terrain_maps.run_pipeline(
                kml_path, folder, api_key, log=log, progress=progress, **params)

            # Remove big intermediates, then bundle the deliverables
            for name in ("dem.tif", "upscaled_dem.tif"):
                (folder / name).unlink(missing_ok=True)
            files = [p for p in result["maps"] + result["rasters"] if Path(p).exists()]
            with zipfile.ZipFile(folder / "terrain_maps.zip", "w", zipfile.ZIP_DEFLATED) as zf:
                for p in files:
                    zf.write(p, Path(p).name)
            with jobs_lock:
                job["maps"] = [Path(p).name for p in result["maps"] if Path(p).exists()]
                job["rasters"] = [Path(p).name for p in result["rasters"] if Path(p).exists()]
                job["status"] = "done"
        except Exception as exc:  # noqa: BLE001 - report everything to the user
            with jobs_lock:
                job["error"] = str(exc).replace(api_key, "***") or exc.__class__.__name__
                job["status"] = "error"


# ---------------------------------------------------------------------------
# routes
# ---------------------------------------------------------------------------
@app.get("/")
def index():
    return render_template("index.html", dem_types=sorted(DEM_TYPES))


@app.post("/api/jobs")
def create_job():
    form = request.form
    api_key = (form.get("api_key") or "").strip()
    if not api_key:
        return jsonify(error="Enter your OpenTopography API key."), 400

    kml = request.files.get("kml")
    if kml is None or not kml.filename:
        return jsonify(error="Choose a boundary .kml file."), 400
    if not kml.filename.lower().endswith(".kml"):
        return jsonify(error="The boundary must be a .kml file (KMZ is not supported)."), 400

    dem_type = form.get("dem_type", "SRTMGL1")
    if dem_type not in DEM_TYPES:
        return jsonify(error="Unknown DEM type."), 400

    try:
        params = dict(
            spring_name=(form.get("spring_name") or "").strip()[:80],
            interval=_number(form, "interval", 5.0, 0.5, 500),
            buffer=_number(form, "buffer", 200.0, 0, 2000),
            subdivide=_number(form, "subdivide", 5, 1, 10, integer=True),
            zoom=_number(form, "zoom", 18, 10, 19, integer=True),
            dem_type=dem_type,
            use_basemap=form.get("basemap") == "on",
        )
    except ValueError as exc:
        return jsonify(error=str(exc)), 400

    _cleanup_old_jobs()
    job_id = uuid.uuid4().hex
    folder = JOB_ROOT / job_id
    folder.mkdir(parents=True)
    kml_path = folder / "input.kml"
    kml.save(kml_path)

    with jobs_lock:
        jobs[job_id] = {
            "created": time.time(), "status": "queued", "progress": 0,
            "log": [], "error": None, "maps": [], "rasters": [],
            "zip_name": _slug(params["spring_name"]) + "_terrain_maps.zip",
        }
    threading.Thread(target=_worker, args=(job_id, folder, kml_path, api_key, params),
                     daemon=True).start()
    return jsonify(job_id=job_id)


@app.get("/api/jobs/<job_id>")
def job_status(job_id):
    if not JOB_ID_RE.match(job_id):
        abort(404)
    with jobs_lock:
        job = jobs.get(job_id)
        if job is None:
            return jsonify(error="Job not found (it may have expired)."), 404
        since = request.args.get("since", default=0, type=int)
        return jsonify(
            status=job["status"], progress=job["progress"], error=job["error"],
            log=job["log"][since:], log_len=len(job["log"]),
            maps=job["maps"], rasters=job["rasters"],
        )


@app.get("/jobs/<job_id>/files/<name>")
def job_file(job_id, name):
    if not JOB_ID_RE.match(job_id) or name not in DOWNLOADABLE:
        abort(404)
    folder = JOB_ROOT / job_id
    if not (folder / name).is_file():
        abort(404)
    as_attachment = request.args.get("dl") == "1"
    download_name = None
    if name == "terrain_maps.zip":
        as_attachment = True
        with jobs_lock:
            download_name = jobs.get(job_id, {}).get("zip_name")
    return send_from_directory(folder, name, as_attachment=as_attachment,
                               download_name=download_name)


if __name__ == "__main__":
    app.run(host=os.environ.get("HOST", "127.0.0.1"),
            port=int(os.environ.get("PORT", "8080")), threaded=True)
