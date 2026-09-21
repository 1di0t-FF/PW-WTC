"""Compute the four detection quantities per scene (the statistics layer).

    python compute_statistics.py                 # all scenes in config.json
    python compute_statistics.py E2-PCS E1-JAM-SP

Reads  input/<MODE>/<scene>/   (C/L/MP per satellite, clock)
       input/<MODE>/<scene>/thresholds.csv (CD2_P_center only)
Writes results/<MODE>/<scene>/satellites/Gxx.csv   (segment, end_index,
       PW_WTC, E_CCDC, CD2_P)  and  clkbm.csv (clock, baseline, bounds)

"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat, savemat

ROOT = Path(__file__).resolve().parent
PW_WINDOW = 600
STAT_LENGTH = 20
CLK_BAND = 15.0        # clkBM alarm half-band, ns/s
CLK_CALIB = 600        # valid AU clock samples behind the baseline
FS, FREQ_LIMITS, VOICES, ALPHA = 1.0, (0.002, 0.020), 12, 0.5
MATLAB = r"D:\Program Files\MATLAB\R2024a\bin\matlab.exe"


def sliding_sum(values, length):
    """Sum of the trailing `length` samples; NaN while the window is incomplete."""
    out = np.full(len(values), np.nan)
    for end in range(length - 1, len(values)):
        window = values[end - length + 1:end + 1]
        if np.isfinite(window).all():
            out[end] = window.sum()
    return out


def ecc_residual(code, carrier):
    """Doppler-consistency residual r = -dL - 1540 * (-dC*f/c), at the pair end."""
    ok = np.isfinite(code[1:]) & np.isfinite(code[:-1]) \
        & np.isfinite(carrier[1:]) & np.isfinite(carrier[:-1])
    r = np.full(len(code), np.nan)
    r[1:][ok] = (carrier[:-1] - carrier[1:])[ok] \
        - 1540.0 * (-(code[1:] - code[:-1]) * 1.023e6 / 299792458.0)[ok]
    return r


def second_diff(code):
    """Second difference C[i] - 2C[i-1] + C[i-2], stored at i."""
    out = np.full(len(code), np.nan)
    ok = np.isfinite(code[2:]) & np.isfinite(code[1:-1]) & np.isfinite(code[:-2])
    out[2:][ok] = (code[2:] - 2 * code[1:-1] + code[:-2])[ok]
    return out


def read_scene(scene):
    """Scene definition plus per-(satellite, segment) input streams."""
    cfg = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))[scene]
    mode = cfg["mode"]
    scene_dir = ROOT / "input" / mode / scene
    segments = ["au", "sp"] if mode == "JAM-SP" else ["stream"]
    streams = {}
    for sat in cfg["satellites"]:
        for seg in segments:
            tag = f"_{seg}" if mode == "JAM-SP" else ""
            test = pd.read_csv(scene_dir / f"{sat}{tag}.csv", encoding="utf-8-sig")
            base = pd.read_csv(scene_dir / f"{sat}_base{tag}.csv", encoding="utf-8-sig")
            clock = pd.read_csv(scene_dir / f"receiver_clock{tag}.csv",
                                encoding="utf-8-sig")["DeltaClkB_nsps"].to_numpy(float)
            streams[(sat, seg)] = dict(C=test.C.to_numpy(float), L=test.L.to_numpy(float),
                                       MP=test.MP.to_numpy(float),
                                       base=base.MP.to_numpy(float), clock=clock)
    return cfg, streams


def pw_wtc(streams):
    """Per-(satellite, segment) gamma series from one MATLAB batch."""
    windows, positions = [], []
    for (sat, seg), s in streams.items():
        for end in range(PW_WINDOW - 1, len(s["MP"])):
            x = s["base"][end - PW_WINDOW + 1:end + 1]
            y = s["MP"][end - PW_WINDOW + 1:end + 1]
            if np.isfinite(x).all() and np.isfinite(y).all():
                windows.append((x, y))
                positions.append((sat, seg, end))
    with tempfile.TemporaryDirectory(prefix="pwwtc_") as temp:
        in_mat, out_mat = Path(temp) / "in.mat", Path(temp) / "out.mat"
        savemat(in_mat, {"x_windows": np.asarray([w[0] for w in windows]),
                         "y_windows": np.asarray([w[1] for w in windows]),
                         "Fs": np.array([[FS]]), "freq_limits": np.array([FREQ_LIMITS]),
                         "voices_per_oct": np.array([[VOICES]]), "wavelet_type": "amor",
                         "power_weight_alpha": np.array([[ALPHA]])}, do_compression=True)
        command = "addpath('{}'); run_pwwtc_batch('{}','{}')".format(
            (ROOT / "matlab").as_posix(), in_mat.as_posix(), out_mat.as_posix())
        print(f"  PW-WTC MATLAB batch: {len(windows)} windows", flush=True)
        subprocess.run([MATLAB, "-batch", command], check=True)
        values = np.asarray(loadmat(out_mat)["gamma"], float).reshape(-1)
    rows = [{"satellite": sat, "segment": seg, "end_index": end, "PW_WTC": value}
            for (sat, seg, end), value in zip(positions, values)]
    out = pd.DataFrame(rows)
    frames = []
    for (sat, seg), s in streams.items():
        grid = out[(out.satellite == sat) & (out.segment == seg)] \
            .set_index("end_index").reindex(range(len(s["MP"])))
        frames.append(pd.DataFrame({"satellite": sat, "segment": seg,
                                    "end_index": range(len(s["MP"])),
                                    "PW_WTC": grid.PW_WTC.to_numpy()}))
    full = pd.concat(frames)
    return {k: full[(full.satellite == k[0]) & (full.segment == k[1])]
               .set_index("end_index").PW_WTC.to_numpy(float)
            for k in streams}


def run_scene(scene):
    cfg, streams = read_scene(scene)
    mode = cfg["mode"]
    out_dir = ROOT / "results" / mode / scene
    thr = pd.read_csv(ROOT / "input" / mode / scene / "thresholds.csv").set_index("satellite")
    gamma = pw_wtc(streams)
    au = "au" if mode == "JAM-SP" else "stream"
    clock_au = streams[(cfg["satellites"][0], au)]["clock"][:900]
    baseline = float(np.nanmean(clock_au[np.isfinite(clock_au)][:CLK_CALIB]))
    lower, upper = baseline - CLK_BAND, baseline + CLK_BAND
    (out_dir / "satellites").mkdir(parents=True, exist_ok=True)
    # Receiver-level clock: one row set per segment, not per satellite.
    clock_rows = []
    for (s, seg), st in streams.items():
        if s != cfg["satellites"][0]:
            continue
        clock_rows.append(pd.DataFrame({
            "segment": seg, "end_index": range(len(st["clock"])),
            "DeltaClkB_nsps": st["clock"], "Baseline_nsps": baseline,
            "LowerBound_nsps": lower, "UpperBound_nsps": upper}))
    for sat in cfg["satellites"]:
        center = float(thr.loc[sat, "CD2_P_center"])
        stat_rows = []
        for (s, seg), st in streams.items():
            if s != sat:
                continue
            stat_rows.append(pd.DataFrame({
                "segment": seg, "end_index": range(len(st["MP"])),
                "PW_WTC": gamma[(s, seg)],
                "E_CCDC": sliding_sum(ecc_residual(st["C"], st["L"]) ** 2, STAT_LENGTH),
                "CD2_P": sliding_sum((second_diff(st["C"]) - center) ** 2, STAT_LENGTH)}))
        pd.concat(stat_rows).to_csv(out_dir / "satellites" / f"{sat}.csv", index=False)
    pd.concat(clock_rows).to_csv(out_dir / "clkbm.csv", index=False)
    print(f"[{scene}] statistics written")


if __name__ == "__main__":
    config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    for scene in (sys.argv[1:] or list(config)):
        run_scene(scene)
