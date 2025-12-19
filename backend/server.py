from __future__ import annotations
from copyreg import pickle
import json
from flask import Flask, request, jsonify, send_file, abort
from io import BytesIO
from flask_cors import CORS
import os, sys, signal, shutil
from pathlib import Path
import geopandas as gpd
from flask import send_from_directory, abort
import osmnx as ox
import matplotlib
import pandas as pd
import numpy as np
import cv2
import subprocess
import rasterio
import threading
import json as jsonlib

worker_proc = None
worker_lock = threading.Lock()
worker_python_exe = None

app = Flask(__name__)
CORS(app)

DATA_DIR = Path("data")        
OUT_DIR  = Path("data/served")
vector_subdir = Path(OUT_DIR / "vector")
raster_subdir = Path(OUT_DIR / "raster")
metric_subdir = Path(OUT_DIR / "metric")
OUT_DIR.mkdir(parents=True, exist_ok=True)
vector_subdir.mkdir(parents=True, exist_ok=True)
raster_subdir.mkdir(parents=True, exist_ok=True)
metric_subdir.mkdir(parents=True, exist_ok=True)

def start_worker():
    global worker_proc, worker_python_exe

    if worker_proc is not None and worker_proc.poll() is None:
        # already running
        return

    project_dir = Path(__file__).parent
    python_exe = project_dir / "envs" / ("python.exe" if os.name == "nt" else "bin/python")

    if not python_exe.exists():
        raise RuntimeError(f"Python interpreter not found at: {python_exe}")

    worker_python_exe = python_exe

    worker_proc = subprocess.Popen(
        [str(python_exe), "-u", "python_worker.py"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=str(project_dir),
        env={**os.environ, "PYTHONPATH": str(project_dir) + os.pathsep + os.environ.get("PYTHONPATH", "")},
        bufsize=1,  # line-buffered
    )

    print("[WORKER] Started python_worker process PID:", worker_proc.pid)

def send_code_to_worker(code: str) -> dict:
    """
    Sends code to the persistent worker and returns a dict:
    { "ok": bool, "stdout": str, "stderr": str }
    """
    global worker_proc

    if worker_proc is None or worker_proc.poll() is not None:
        start_worker()

    with worker_lock:
        req = {"code": code}
        line = jsonlib.dumps(req) + "\n"

        assert worker_proc.stdin is not None
        worker_proc.stdin.write(line)
        worker_proc.stdin.flush()

        assert worker_proc.stdout is not None
        resp_line = worker_proc.stdout.readline()
        if not resp_line:
            raise RuntimeError("Worker process terminated unexpectedly")

    resp = jsonlib.loads(resp_line)
    return resp

# Remove cached outputs on exit
def cleanup_served():
    try:
        out_resolved = OUT_DIR.resolve()
        data_resolved = DATA_DIR.resolve()
        if data_resolved in out_resolved.parents and out_resolved.name == "served":
            if OUT_DIR.exists():
                shutil.rmtree(OUT_DIR, ignore_errors=True)
                print(f"[CLEANUP] Removed {OUT_DIR}")
        else:
            print(f"[CLEANUP] Refusing to remove unexpected path: {OUT_DIR}")
    except Exception as e:
        app.logger.error(f"[CLEANUP] Failed: {e}")

def register_cleanup_handlers():
    import atexit
    atexit.register(cleanup_served)

    def _handler(sig, frame):
        cleanup_served()
        sys.exit(0)

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _handler)
        except Exception:
            pass

if os.environ.get("WERKZEUG_RUN_MAIN") == "true" or not app.debug:
    register_cleanup_handlers()

def resolve_feather_path(datafile: str, tag: str) -> Path:
    feather_filename = f"osm/processed/{datafile}/{tag}.feather"
    return DATA_DIR / feather_filename

def load_roi_mask(roi: dict) -> tuple[str, object]:
    rtype = roi.get("type")
    val = roi.get("value")
    if rtype == "bbox":
        xmin, ymin, xmax, ymax = map(float, val)
        return "bbox", (xmin, ymin, xmax, ymax)
    
    elif rtype == "geojson":
        roi_path = Path(val)
        roi_path = DATA_DIR / str(val)

        # Try this later in notebook and then add.. 

        # mask_gdf = gpd.read_file(roi_path)
        # mask = mask_gdf.union_all()
        # return "mask", gpd.GeoDataFrame(geometry=[mask], crs=mask_gdf.crs)

def crop_gdf(gdf: gpd.GeoDataFrame, roi_dict: dict) -> gpd.GeoDataFrame:
    roi_kind, roi = load_roi_mask(roi_dict)

    if roi_kind == "bbox":
        xmin, ymin, xmax, ymax = roi
        return gdf.cx[xmin:xmax, ymin:ymax]
    else:
        mask_gdf = roi  
        if mask_gdf.crs is None:
            mask_gdf = mask_gdf.set_crs(4326, allow_override=True)

        return gpd.clip(gdf, mask_gdf)
    
def select_features(gdf: gpd.GeoDataFrame, features: list[str]) -> gpd.GeoDataFrame:
    existing = [f for f in features if f in gdf.columns]
    cols = existing + (["geometry"] if "geometry" in gdf.columns else [])
    
    return gdf[cols]

@app.get("/api/list-rasters/<Id>")
def list_rasters(Id: str):
    """
    Return a JSON list of PNG raster tiles inside:
    data/served/raster/<Id>/
    """
    folder = raster_subdir / Id

    try:
        folder.resolve().relative_to(raster_subdir.resolve())
    except Exception:
        return jsonify({"error": "Invalid raster folder"}), 403

    if not folder.exists() or not folder.is_dir():
        return jsonify([]), 200   # return empty list

    files = [
        f.name
        for f in folder.iterdir()
        if f.is_file() and f.suffix.lower() == ".png"
    ]

    return jsonify(files), 200

@app.get("/generated/raster/<path:filename>")
def serve_raster(filename: str):
    allowed_exts = {"png", "tif", "tiff"}

    try:
      ext = filename.rsplit(".", 1)[1].lower()
    except IndexError:
      abort(404)

    if ext not in allowed_exts:
        abort(404)

    full_path = raster_subdir / filename
    try:
        full_path.resolve().relative_to(raster_subdir.resolve())
    except Exception:
        abort(403)  # Forbidden

    directory = full_path.parent
    file = full_path.name

    if ext == "png":
        mimetype = "image/png"
    else:  # tif / tiff
        mimetype = "image/tiff"

    return send_from_directory(
        directory,
        file,
        mimetype=mimetype,
        conditional=True,
    )

@app.get("/generated/vector/<path:filename>")
def serve_vector(filename: str):
    if not filename.lower().endswith(".geojson"):
        abort(404)

    return send_from_directory(
        vector_subdir,
        filename,
        mimetype="application/geo+json",
        conditional=True
    )

@app.post('/api/ingest-physical-layer')
def ingest_physical_layer():

    payload = request.get_json(silent=True) or {}
    pl = payload
    problems = []
    
    pl_id = pl.get("id")
    datafile = pl.get("datafile")
    roi = pl.get("region_of_interest") or {}

    print(f"Processing physical layer ID: {pl_id} with datafile: {datafile}")

    for lyr in (pl.get("layers") or []):
        tag = lyr.get("tag")
        features = lyr.get("features") or []
        try:
            src_path = resolve_feather_path(str(datafile), str(tag))
            if not src_path.is_file():
                raise FileNotFoundError(f"Missing source: {src_path}")
            print(f"    Loading data from: {src_path}")
            gdf = gpd.read_feather(src_path)

            gdf_cut = crop_gdf(gdf, roi)
            gdf_out = select_features(gdf_cut, features)

            out_name = f"vector/{pl_id}_{tag}.geojson"
            out_path = OUT_DIR / out_name
            
            gdf_out.to_file(out_path, driver="GeoJSON")

        except Exception as e:
            error_msg = f"[ERROR] Layer {pl_id}:{tag} → {type(e).__name__}: {e}"
            print(error_msg)

            problems.append({
                "physical_layer_id": pl_id,
                "tag": tag,
                "error": str(e)
            })

    return jsonify({
        "status": "success" if not problems else "partial",
        "problems": problems
    }), 200 if not problems else 207  

@app.route("/api/update-data-layer", methods=["POST"])
def update_data_layer():
    data = request.get_json()
    ref = data["ref"]
    # tag = data["tag"]
    geojson = data["geojson"]

    filename = f"vector/{ref}.geojson"
    filepath = OUT_DIR / filename

    with open(filepath, "w") as f:
        json.dump(geojson, f)

    return jsonify({"status": "success"}), 200

@app.post("/api/run-python")
def run_python():
    payload = request.get_json() or {}
    code = payload.get("code", "")

    print("Received code to run:\n", code)

    try:
        resp = send_code_to_worker(code)
        # resp: {"ok": bool, "stdout": "...", "stderr": "..."}

        return jsonify({
            "stdout": resp.get("stdout", ""),
            "stderr": resp.get("stderr", ""),
            "returncode": 0 if resp.get("ok") else 1,
        }), 200

    except Exception as e:
        # If something goes really wrong (worker dead, etc.)
        return jsonify({
            "stdout": "",
            "stderr": f"Worker error: {e}",
            "returncode": -1,
        }), 200
    
COLORMAPS = {
    "viridis": matplotlib.colormaps.get_cmap("viridis"),
    "reds": matplotlib.colormaps.get_cmap("Reds"),
    "greens": matplotlib.colormaps.get_cmap("Greens"),
    "blues": matplotlib.colormaps.get_cmap("Blues"),
    "grays":   matplotlib.colormaps.get_cmap("Greys")
}

@app.get("/generated/raster/<ref>/<name>")
def get_colormapped_tile(ref, name):
    path = raster_subdir / ref / name

    cmap_name = (request.args.get("cmap") or "").lower()
    cmap = COLORMAPS.get(cmap_name)

    # If no/unknown colormap → return original PNG
    if cmap is None:
        return send_file(path, mimetype="image/png")

    # Read image (gray / BGR / BGRA)
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        abort(404)

    # ---- extract grayscale 0–255 ----
    if img.ndim == 2:
        gray = img.astype(np.float32)
        alpha = None
    elif img.shape[2] == 4:
        gray = img[:, :, 0].astype(np.float32)
        alpha = img[:, :, 3]
    else:
        gray = img[:, :, 0].astype(np.float32)
        alpha = None

    # ---- normalize FIXED 0–255 ----
    t = np.clip(gray / 255.0, 0.0, 1.0)

    # ---- apply matplotlib colormap ----
    rgba = (cmap(t) * 255).astype(np.uint8)  # H x W x 4

    # ---- preserve alpha if present ----
    if alpha is not None:
        rgba[:, :, 3] = alpha

    rgba = rgba[..., [2, 1, 0, 3]]  # RGBA -> BGRA for OpenCV

    ok, buf = cv2.imencode(".png", rgba)
    if not ok:
        abort(500)

    return send_file(BytesIO(buf.tobytes()), mimetype="image/png")
    
def diff_dirs(dir1_, dir2_):
    dir1 = Path(raster_subdir, dir1_)
    dir2 = Path(raster_subdir, dir2_)
    output_path = Path(raster_subdir, f"{dir1_}_minus_{dir2_}")

    if output_path.exists():
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    for png1 in sorted(dir1.glob("*.png")):
        png2 = dir2 / png1.name
        if not png2.exists():
            print(f"Skipping {png1.name}: not found in {dir2}")
            continue

        # Read as grayscale (single channel 0..255)
        gray1 = cv2.imread(str(png1), cv2.IMREAD_GRAYSCALE)
        gray2 = cv2.imread(str(png2), cv2.IMREAD_GRAYSCALE)

        if gray1 is None or gray2 is None:
            print(f"Skipping {png1.name}: failed to read one of the images")
            continue

        # absolute difference in 0..255 (uint8)
        diff_u8 = cv2.absdiff(gray2, gray1)

        # Save as grayscale PNG (no colormap baked in)
        cv2.imwrite(str(output_path / png1.name), diff_u8)

    return output_path

@app.route("/api/diff-png", methods=["POST"])
def api_diff_png():
    data = request.get_json(force=True)
    dir1 = data.get("dir1")
    dir2 = data.get("dir2")
    # colormap = data.get("colormap", "Reds")

    if not dir1 or not dir2:
        return jsonify({"error": "dir1 and dir2 are required"}), 400

    out_dir = diff_dirs(dir1, dir2)
    return jsonify({
        "status": "ok",
        "output_dir": str(out_dir),
    })

def diff_tif_files(tif1_name_, tif2_name_):

    tif1_name = tif1_name_ + ".tif"
    tif2_name = tif2_name_ + ".tif"

    tif1_path = Path(raster_subdir, tif1_name)
    tif2_path = Path(raster_subdir, tif2_name)

    out_name = f"{tif1_name_}_minus_{tif2_name_}.tif"
    out_path = Path(raster_subdir, out_name)

    with rasterio.open(tif1_path) as r1, rasterio.open(tif2_path) as r2:
        arr1 = r1.read(1).astype("float32")
        arr2 = r2.read(1).astype("float32")

        # Difference: scenario2 - scenario1
        diff = np.abs(arr2 - arr1)

        profile = r1.profile
        profile.update(
            dtype="float32",
            count=1,
        )

        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(diff, 1)

    return out_path

@app.route("/api/diff-tif", methods=["POST"])
def api_diff_tif():
    data = request.get_json(force=True)
    tif1 = data.get("tif1")
    tif2 = data.get("tif2")

    if not tif1 or not tif2:
        return jsonify({"error": "tif1 and tif2 are required"}), 400

    out_path = diff_tif_files(tif1, tif2)

    return jsonify({
        "status": "ok",
        "output_tif": str(out_path),
    })

@app.post("/api/comparison-view")
def comparison_view():
    data = request.get_json(force=True)

    print("Received comparison view request:", data)
    key = data.get("key", [])
    metric = data.get("metric", "")
    encoding = data.get("encoding", "")
    unit = data.get("unit", "")

    results = {}   # store metric per layer
    for k in key:
        csv_path = metric_subdir / f"{k}.csv"
        df = pd.read_csv(csv_path)

        value = df[metric].iloc[0]
        results[k] = float(value)

        print(f"{k}: {metric} = {value}")

    return jsonify({
        "status": "ok",
        "metric": metric,
        "unit": unit,
        "values": results,
        "encoding": encoding,
    })


if __name__ == '__main__':
    # remove old served before starting
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR, ignore_errors=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    vector_subdir.mkdir(parents=True, exist_ok=True)
    raster_subdir.mkdir(parents=True, exist_ok=True)
    metric_subdir.mkdir(parents=True, exist_ok=True)
    
    # Start the worker up-front (or you can let send_code_to_worker lazily do it)
    try:
        start_worker()
    except Exception as e:
        print("[WORKER] Failed to start:", e)

    app.run(host='0.0.0.0', port=5000, debug=True)