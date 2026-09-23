import argparse
import glob
import json
import os
import subprocess
import sys

import numpy as np

SCHEMA_VERSION = 3

try:
    import h5py
except ImportError:
    sys.exit("h5py is required: pip install h5py")

try:
    import cv2
    def _resize(img, w, h):
        return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
except ImportError:
    from PIL import Image
    def _resize(img, w, h):
        return np.asarray(Image.fromarray(img).resize((w, h), Image.BILINEAR))

MARKER_ROWS = 2


def probe_dims(path):
    """Width, full height and fps of an h264 stream via ffprobe."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,r_frame_rate",
         "-of", "csv=p=0", path],
        capture_output=True, text=True, check=True).stdout.strip()
    w, h, rate = out.split(",")
    num, den = (rate.split("/") + ["1"])[:2]
    fps = float(num) / float(den) if float(den) else 30.0
    return int(w), int(h), fps


def decode_frames(path, full_w, full_h):
    """Yield decoded RGB frames at full height, marker rows included."""
    proc = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-i", path, "-f", "rawvideo",
         "-pix_fmt", "rgb24", "-"],
        stdout=subprocess.PIPE)
    frame_bytes = full_w * full_h * 3
    while True:
        raw = proc.stdout.read(frame_bytes)
        if len(raw) < frame_bytes:
            break
        yield np.frombuffer(raw, np.uint8).reshape(full_h, full_w, 3)
    proc.stdout.close()
    proc.wait()


def decode_marker_u64(row):
    """Decode a 64-bit little-endian value from a marker row."""
    bits = (row[:64, 0].astype(np.uint64) > 128)
    val = np.uint64(0)
    for b in range(64):
        if bits[b]:
            val |= np.uint64(1) << np.uint64(b)
    return int(val)


def load_video(path, out_w, out_h):
    """Load resized frames, wall-clock timestamps and frame ids from an h264 video."""
    full_w, full_h, fps = probe_dims(path)
    img_h = full_h - MARKER_ROWS

    sidecar = os.path.splitext(path)[0] + ".timestamps.csv"
    side_ts = None
    if os.path.exists(sidecar):
        arr = np.genfromtxt(sidecar, delimiter=",", skip_header=1)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        side_ts = arr[:, 1].astype(np.int64)

    frames, ts, fids = [], [], []
    for i, frame in enumerate(decode_frames(path, full_w, full_h)):
        if side_ts is not None and i < len(side_ts):
            wall, fid = int(side_ts[i]), i
        else:
            wall = decode_marker_u64(frame[img_h])
            fid  = decode_marker_u64(frame[img_h + 1])
        img = frame[:img_h]
        if img.shape[:2] != (out_h, out_w):
            img = _resize(img, out_w, out_h)
        frames.append(img)
        ts.append(wall)
        fids.append(fid)

    if not frames:
        return None
    return np.asarray(frames), np.asarray(ts, np.int64), np.asarray(fids, np.int64)


def load_image_log(path, name, out_w, out_h):
    """Load resized frames, wall-clock timestamps and frame ids from an images_<name>.hdf5 log."""
    side = read_timestamps(os.path.splitext(path)[0] + ".timestamps.csv")
    with h5py.File(path, "r") as f:
        ds = f[f"observations/images/{name}"]
        n = ds.shape[0]
        native = ds.shape[1:3]
        frames = np.empty((n, out_h, out_w, 3), np.uint8)
        for i in range(0, n, 64):
            for j, img in enumerate(ds[i:i + 64]):
                frames[i + j] = img if img.shape[:2] == (out_h, out_w) else _resize(img, out_w, out_h)
        if side is not None:
            ts, fids = side[:n, 1], side[:n, 3]
        else:
            ts, fids = f["observations/timestamp_ns"][:n].astype(np.int64), np.arange(n)
    if n == 0:
        return None
    return frames, ts.astype(np.int64), fids.astype(np.int64), (native[1], native[0])


def read_timestamps(path):
    """Integer array from a camera timestamps csv, or None."""
    if not os.path.exists(path):
        return None
    arr = np.genfromtxt(path, delimiter=",", skip_header=1, dtype=np.int64)
    return arr.reshape(1, -1) if arr.ndim == 1 else arr


def load_csv(path):
    """Load a telemetry csv as header, data and wall-clock timestamps."""
    with open(path) as f:
        header = f.readline().strip().split(";")
    data = np.genfromtxt(path, delimiter=";", skip_header=1)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    wall = data[:, header.index("wall_clock_ns")].astype(np.int64)
    return header, data, wall


def col_group(header, data, prefix, count):
    """Stack columns prefix0..prefixN-1 into [N, count], or None."""
    names = [f"{prefix}{i}" for i in range(count)]
    if not all(n in header for n in names):
        return None
    idx = [header.index(n) for n in names]
    return data[:, idx]


def col_one(header, data, name):
    """Single named column, or None."""
    return data[:, header.index(name)] if name in header else None


def nearest_idx(stream_ts, grid_ts):
    """Index of the nearest stream sample for each grid time."""
    pos = np.searchsorted(stream_ts, grid_ts)
    pos = np.clip(pos, 1, len(stream_ts) - 1)
    left, right = stream_ts[pos - 1], stream_ts[pos]
    return np.where(np.abs(grid_ts - left) <= np.abs(right - grid_ts), pos - 1, pos)


def read_meta(folder):
    """Episode metadata from arm_left_meta.csv."""
    out = {}
    mpath = os.path.join(folder, "arm_left_meta.csv")
    if not os.path.exists(mpath):
        return out
    with open(mpath) as f:
        rows = [ln.strip().split(";") for ln in f if ln.strip()]
    if not rows:
        return out
    hdr = rows[0]
    for r in rows[1:]:
        d = dict(zip(hdr, r))
        if d.get("event") == "episode_config":
            out["seed"] = d.get("seed", "")
            out["mode"] = d.get("mode", "")
            out["color_bin_mapping"] = d.get("color_bin_mapping", "")
        if d.get("event") == "episode_end":
            out["success"] = d.get("color_bin_mapping", "")
    return out


def load_camera_params(folder, params_path):
    """Load camera_params.json from the given or default locations."""
    candidates = []
    if params_path:
        candidates.append(params_path)
    candidates += [
        os.path.join(folder, "camera_params.json"),
        os.path.join(folder, "..", "camera_params.json"),
    ]
    for p in candidates:
        if p and os.path.exists(p):
            with open(p) as f:
                return json.load(f)
    return {}


def _write_eye_dataset(imgs_group, eye_name, eye_frames, stereo_attrs, cam_params, stereo_cam_name):
    """Write one eye of the stereo head camera with its attrs."""
    H, eye_w = eye_frames.shape[1], eye_frames.shape[2]
    ds = imgs_group.create_dataset(
        eye_name, data=eye_frames,
        compression="gzip", compression_opts=4,
        chunks=(1, H, eye_w, eye_frames.shape[3]),
    )
    for k, v in stereo_attrs.items():
        ds.attrs[k] = v // 2 if k == "native_width" else v
    if stereo_cam_name in cam_params:
        cp = cam_params[stereo_cam_name]
        for key in ("fx", "fy", "cx", "cy", "width", "height"):
            if key in cp:
                ds.attrs[key] = cp[key]
        if "T_world_cam" in cp:
            ds.attrs["T_world_cam"] = np.asarray(cp["T_world_cam"], dtype=np.float64)


def convert(folder, out_path, rate, size, cameras, camera_params_path=None):
    """Align all streams of one episode folder onto a fixed-rate grid and write hdf5."""
    out_h, out_w = size
    cams = {}
    cam_native_dims = {}
    for name in cameras:
        vpath = os.path.join(folder, f"video_{name}.h264")
        hpath = os.path.join(folder, f"images_{name}.hdf5")
        if os.path.exists(vpath):
            full_w, full_h, _ = probe_dims(vpath)
            cam_native_dims[name] = (full_w, full_h - MARKER_ROWS)
            v = load_video(vpath, out_w * 2 if name == "head_cam_stereo" else out_w, out_h)
            if v is not None:
                cams[name] = v
        elif os.path.exists(hpath):
            v = load_image_log(hpath, name, out_w, out_h)
            if v is not None:
                cams[name] = v[:3]
                cam_native_dims[name] = v[3]
        else:
            print(f"  [warn] {folder}: camera {name} not found")

    cam_params = load_camera_params(folder, camera_params_path)

    arms, arm_ts = {}, {}
    for arm in ("arm_left", "arm_right"):
        p = os.path.join(folder, f"{arm}.csv")
        if os.path.exists(p):
            arms[arm] = load_csv(p)
            arm_ts[arm] = arms[arm][2]

    head = None
    hp = os.path.join(folder, "head.csv")
    if os.path.exists(hp):
        head = load_csv(hp)

    starts, ends = [], []
    for _, ts, _ in cams.values():
        starts.append(ts[0]); ends.append(ts[-1])
    for ts in arm_ts.values():
        starts.append(ts[0]); ends.append(ts[-1])
    if head is not None:
        starts.append(head[2][0]); ends.append(head[2][-1])
    if not starts:
        print(f"  [skip] {folder}: no streams found")
        return
    t0, t1 = max(starts), min(ends)
    if t1 <= t0:
        print(f"  [skip] {folder}: streams do not overlap")
        return
    dt = int(1e9 / rate)
    grid = np.arange(t0, t1, dt, dtype=np.int64)
    T = len(grid)

    meta = read_meta(folder)

    with h5py.File(out_path, "w") as f:
        f.attrs["schema_version"] = SCHEMA_VERSION
        f.attrs["rate_hz"] = rate
        f.attrs["image_size"] = np.asarray(size)
        f.attrs["episode_id"] = os.path.basename(folder.rstrip("/\\"))
        for k, v in meta.items():
            f.attrs[k] = v

        obs = f.create_group("observations")
        obs.create_dataset("timestamp_ns", data=grid)

        imgs = obs.create_group("images")
        first_cam = True
        for name, (frames, ts, fids) in cams.items():
            sel = nearest_idx(ts, grid)
            selected = frames[sel]

            if name == "head_cam_stereo":
                eye_w = selected.shape[2] // 2
                stereo_attrs = {}
                if name in cam_native_dims:
                    stereo_attrs["native_width"]  = cam_native_dims[name][0]
                    stereo_attrs["native_height"] = cam_native_dims[name][1]
                _write_eye_dataset(imgs, "head_cam_left",  selected[:, :, :eye_w, :],
                                   stereo_attrs, cam_params, name)
                _write_eye_dataset(imgs, "head_cam_right", selected[:, :, eye_w:, :],
                                   stereo_attrs, cam_params, name)
            else:
                ds = imgs.create_dataset(name, data=selected,
                                         compression="gzip", compression_opts=4,
                                         chunks=(1,) + selected.shape[1:])
                if name in cam_native_dims:
                    ds.attrs["native_width"]  = cam_native_dims[name][0]
                    ds.attrs["native_height"] = cam_native_dims[name][1]
                if name in cam_params:
                    cp = cam_params[name]
                    for key in ("fx", "fy", "cx", "cy", "width", "height"):
                        if key in cp:
                            ds.attrs[key] = cp[key]
                    if "T_world_cam" in cp:
                        ds.attrs["T_world_cam"] = np.asarray(cp["T_world_cam"], dtype=np.float64)

            if first_cam:
                obs.create_dataset("frame_id", data=fids[sel])
                first_cam = False

        act = f.create_group("actions")
        for arm, (hdr, data, ts) in arms.items():
            sel = nearest_idx(ts, grid)
            g = obs.create_group(arm)


            for field, n in (("q_", 7), ("dq_", 7), ("tau_J_", 7),
                             ("tau_ext_", 7), ("O_T_EE_", 16), ("O_T_EE_world_", 16),
                             ("F_ext_", 6)):
                grp = col_group(hdr, data, field, n)
                if grp is not None:
                    g.create_dataset(field.rstrip("_"), data=grp[sel])
            gw = col_one(hdr, data, "gripper_width")
            if gw is not None:
                g.create_dataset("gripper_width", data=gw[sel])
            for field in ("state", "cmd_valid", "log_src", "clutch", "authority"):
                col = col_one(hdr, data, field)
                if col is not None:
                    g.create_dataset(field, data=col[sel].astype(np.int64))

            ag = act.create_group(arm)
            for field, n in (("q_cmd_", 7), ("O_T_EE_cmd_", 16), ("O_T_EE_cmd_world_", 16)):
                grp = col_group(hdr, data, field, n)
                if grp is not None:
                    ag.create_dataset(field.rstrip("_"), data=grp[sel])
            gc = col_one(hdr, data, "gripper_cmd")
            if gc is not None:
                ag.create_dataset("gripper_cmd", data=gc[sel])

        if head is not None:
            hdr, data, ts = head
            sel = nearest_idx(ts, grid)
            hg = obs.create_group("head")
            for field, n in (("q_", 2), ("dq_", 2), ("tau_J_", 2), ("q_cmd_", 2)):
                grp = col_group(hdr, data, field, n)
                if grp is not None:
                    hg.create_dataset(field.rstrip("_"), data=grp[sel])
            st = col_one(hdr, data, "state")
            if st is not None:
                hg.create_dataset("state", data=st[sel].astype(np.int64))

    print(f"  [ok] {out_path}  T={T}  cams={list(cams)}  "
          f"dur={(t1 - t0) / 1e9:.2f}s")


def _convert_one(args_tuple):
    """Convert one episode, logging errors instead of raising."""
    folder, out_path, rate, size, cameras, camera_params_path = args_tuple
    try:
        convert(folder, out_path, rate, size, cameras,
                camera_params_path=camera_params_path)
    except Exception as e:
        print(f"  [error] {folder}: {e}")


def main():
    """Convert one episode folder or all of them."""
    ap = argparse.ArgumentParser(description="Episode folder to aligned hdf5.")
    ap.add_argument("path", help="episode folder, or logs root with --all")
    ap.add_argument("--all", action="store_true", help="process every NNN/ folder under path")
    ap.add_argument("--rate", type=float, default=20.0, help="output grid rate (Hz)")
    ap.add_argument("--size", type=int, nargs=2, default=[240, 320], metavar=("H", "W"), help="image size per camera")
    ap.add_argument("--cameras", nargs="+",
                    default=["head_cam_stereo", "wrist_cam_left", "wrist_cam_right", "overview_cam"])
    ap.add_argument("--out", default="episode_policy.hdf5", help="output filename within each folder")
    ap.add_argument("--camera-params", default=None,
                    help="path to camera_params.json; also searched at <episode>/camera_params.json")
    ap.add_argument("--jobs", type=int, default=1, help="parallel worker processes; -1 = all cores")
    ap.add_argument("--overwrite", action="store_true", help="re-convert even if output already exists")
    args = ap.parse_args()

    if args.all:
        folders = sorted(d for d in glob.glob(os.path.join(args.path, "[0-9]" * 3)) if os.path.isdir(d))
    else:
        folders = [args.path]

    work = []
    for folder in folders:
        out_path = os.path.join(folder, args.out)
        if os.path.exists(out_path):
            if not args.overwrite:
                print(f"  [skip] {folder}: {args.out} already exists")
                continue
            print(f"  [overwrite] {folder}: re-converting")
        work.append((folder, out_path, args.rate, tuple(args.size), args.cameras, args.camera_params))

    if not work:
        print("Nothing to convert.")
        return

    import multiprocessing
    n_jobs = args.jobs if args.jobs > 0 else multiprocessing.cpu_count()
    n_jobs = min(n_jobs, len(work))

    if n_jobs == 1:
        for item in work:
            print(f"Converting {item[0]} ...")
            _convert_one(item)
    else:
        from concurrent.futures import ProcessPoolExecutor
        print(f"Converting {len(work)} episodes with {n_jobs} workers ...")
        with ProcessPoolExecutor(max_workers=n_jobs) as pool:
            pool.map(_convert_one, work)


if __name__ == "__main__":
    main()
