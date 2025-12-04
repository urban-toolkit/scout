from __future__ import annotations
from copyreg import pickle
# import gzip
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
import os, sys, signal, shutil
from pathlib import Path
import geopandas as gpd
# from shapely.geometry import Polygon
from flask import send_from_directory, abort
from convert_to_raster import convert_raster
# from deep_umbra import run_shadow_model
# from download_data import download_osm_data, extract_roads, extract_buildings
import osmnx as ox
import matplotlib
# import pickle, gzip
import pandas as pd
import numpy as np
import cv2

from weather_routing import *
import subprocess
# import tempfile

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

    # -u for unbuffered so we can get output immediately
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

    # Ensure only one thread talks to the worker at a time
    with worker_lock:
        req = {"code": code}
        line = jsonlib.dumps(req) + "\n"

        # send
        assert worker_proc.stdin is not None
        worker_proc.stdin.write(line)
        worker_proc.stdin.flush()

        # receive one line
        assert worker_proc.stdout is not None
        resp_line = worker_proc.stdout.readline()
        if not resp_line:
            # worker died unexpectedly
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

# Utility functions
def resolve_feather_path(datafile: str, tag: str) -> Path:
    feather_filename = f"{datafile}/{tag}.feather"
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

@app.get("/api/list-rasters/<plId>")
def list_rasters(plId: str):
    """
    Return a JSON list of PNG raster tiles inside:
    data/served/raster/<plId>/
    """
    # Resolve folder safely
    folder = raster_subdir / plId

    # Security: ensure path is inside raster_subdir
    try:
        folder.resolve().relative_to(raster_subdir.resolve())
    except Exception:
        return jsonify({"error": "Invalid raster folder"}), 403

    # Check folder exists
    if not folder.exists() or not folder.is_dir():
        return jsonify([]), 200   # return empty list

    # Collect *.png files
    files = [
        f.name
        for f in folder.iterdir()
        if f.is_file() and f.suffix.lower() == ".png"
    ]

    return jsonify(files), 200

@app.get("/generated/raster/<path:filename>")
def serve_raster(filename: str):
    # Allow only specific raster extensions
    allowed_exts = {"png", "tif", "tiff"}

    # Get extension (everything after last dot)
    try:
      ext = filename.rsplit(".", 1)[1].lower()
    except IndexError:
      abort(404)

    if ext not in allowed_exts:
        abort(404)

    # Resolve safe absolute path (prevents directory traversal)
    full_path = raster_subdir / filename
    try:
        full_path.resolve().relative_to(raster_subdir.resolve())
    except Exception:
        abort(403)  # Forbidden

    directory = full_path.parent
    file = full_path.name

    # Pick correct mimetype
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
        print(lyr)
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

            # if(tag == "roads"):
            #     roi_kind, roi_ = load_roi_mask(roi)
            #     if roi_kind == "bbox":
            #         xmin, ymin, xmax, ymax = roi_
            #         with gzip.open("./data/%s/roads.pkl.gz" % datafile, "rb") as f:
            #             G = pickle.load(f)

            #         nodes, _ = ox.graph_to_gdfs(G, nodes=True, edges=True, fill_edge_geometry=False)
            #         mask = (
            #             (nodes["y"] <= ymax) & (nodes["y"] >= ymin) &
            #             (nodes["x"] <= xmax) & (nodes["x"] >= xmin)
            #         )
            #         node_ids = nodes.loc[mask].index
            #         G_crop = G.subgraph(node_ids).copy()

            #         with gzip.open("%s/vector/%s_roads.pkl.gz" % (OUT_DIR, pl_id), "wb") as f:
            #             pickle.dump(G_crop, f, protocol=pickle.HIGHEST_PROTOCOL)

            #         print(f"Saved cropped road graph to: {OUT_DIR}/vector/{pl_id}_roads.pkl.gz")

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

@app.route("/api/update-physical-layer", methods=["POST"])
def update_physical_layer():
    data = request.get_json()
    pl_id = data["physicalLayerRef"]
    tag = data["tag"]
    geojson = data["geojson"]

    filename = f"vector/{pl_id}_{tag}.geojson"
    filepath = OUT_DIR / filename

    with open(filepath, "w") as f:
        json.dump(geojson, f)

    return jsonify({"status": "success"}), 200

@app.route("/api/convert-to-raster", methods=["POST"])
def convert_to_raster():
    # Using code node instead of using this api from grammar!
    data = request.get_json()
    pl_id = data["physical_layer"]["ref"]
    id = data["id"]
    tag = data["layer"]["tag"]
    feature = data["layer"]["feature"]
    zoom = data["zoom"]

    convert_raster(pl_id, tag, feature, zoom, id)

    return jsonify({"status": "success"}), 200

@app.route('/weather', methods=["POST"])
def calculate_weather_aware_route():
    data = request.get_json()
    datafile = data["city"]
    origin = data["origin"]
    destination = data["destination"]
    bbox = data["bbox"] # Bounding box, assuming its sent as [ymax, ymin, xmax, xmin]
    map_view_mode = data["map_view_mode"]
    K_variable_paths = data["paths"]
    weather_conditions = data["weather"]
    weather_weights = data["weights"]
    time = data["time"]
    
    
    route_coords = calculate_weather_route(datafile, 
                            origin, 
                            destination,
                            bbox, 
                            map_view_mode,
                            K_variable_paths,
                            weather_conditions,
                            weather_weights,
                            time)


    return jsonify({"route_coords": route_coords}), 200

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
    
def diff_colormap_dirs(dir1_, dir2_, colormap='Reds'):
    dir1 = Path(raster_subdir, dir1_)
    dir2 = Path(raster_subdir, dir2_)
    output_path = Path(raster_subdir, dir1_ + "_minus_" + dir2_)

    if output_path.exists():
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    for png1 in sorted(dir1.glob("*.png")):
        png2 = dir2 / png1.name
        if not png2.exists():
            print(f"Skipping {png1.name}: not found in {dir2}")
            continue
        
        img1 = cv2.imread(str(png1))
        img2 = cv2.imread(str(png2))

        if img1 is None or img2 is None:
            print(f"Skipping {png1.name}: failed to read one of the images")
            continue

        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        diff = cv2.absdiff(gray2, gray1).astype(np.float32)

        dmin, dmax = diff.min(), diff.max()
        if dmax > dmin:
            diff_norm = (diff - dmin) / (dmax - dmin)
        else:
            diff_norm = np.zeros_like(diff)

        cmap = matplotlib.colormaps[colormap]
        rgba = cmap(diff_norm)
        rgb = (rgba[:, :, :3] * 255).astype("uint8")

        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(output_path / png1.name), bgr)

    return output_path

@app.route("/api/diff-png", methods=["POST"])
def api_diff_png():
    data = request.get_json(force=True)
    dir1 = data.get("dir1")
    dir2 = data.get("dir2")
    colormap = data.get("colormap", "Reds")

    if not dir1 or not dir2:
        return jsonify({"error": "dir1 and dir2 are required"}), 400

    out_dir = diff_colormap_dirs(dir1, dir2, colormap)
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
    data = request.get_json(force=True)   # this is your { key: [...], metric: "...", encoding: "..." }

    print("Received comparison view request:", data)
    key = data.get("key", [])
    metric = data.get("metric", "")
    encoding = data.get("encoding", "")
    unit = data.get("unit", "")

    results = {}   # store metric per layer
    for k in key:
        # look for a csv in metric_subdir with name k + ".csv"
        csv_path = metric_subdir / f"{k}.csv"
        # read csv to get the metric value
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