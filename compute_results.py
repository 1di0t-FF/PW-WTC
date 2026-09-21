"""Compute detection results from stored statistics (the decision layer).

    python compute_results.py                 # all scenes in config.json
    python compute_results.py E2-PCS E1-JAM-SP

Reads  results/<MODE>/<scene>/satellites/Gxx.csv and clkbm.csv
       (produced by compute_statistics.py)
       input/<MODE>/<scene>/thresholds.csv
Writes results/<MODE>/<scene>/detection_rates.csv

A position counts when all four quantities are finite there; detection is
reported for the SP segment only (FCS/PCS: end_index 900..1799 of the
stream; JAM-SP: the sp segment, 301 window-complete positions per
satellite).  Alarms: PW_WTC < PW_WTC_threshold, E_CCDC > E_CCDC_threshold,
CD2_P > CD2_P_threshold, clock outside [LowerBound_nsps, UpperBound_nsps].
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
METHODS = ["PW-WTC", "E-CCDC", "CD2-P", "clkBM"]


def run_scene(scene):
    cfg = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))[scene]
    mode = cfg["mode"]
    thr = pd.read_csv(ROOT / "input" / mode / scene / "thresholds.csv").set_index("satellite")
    clock = pd.read_csv(ROOT / "results" / mode / scene / "clkbm.csv")
    sp = "sp" if mode == "JAM-SP" else "stream"
    sp_start = 0 if mode == "JAM-SP" else 900
    rows = []
    for sat in cfg["satellites"]:
        st = pd.read_csv(ROOT / "results" / mode / scene / "satellites" / f"{sat}.csv")
        st = st[st.segment == sp].reset_index(drop=True)
        cl = clock[clock.segment == sp].reset_index(drop=True)
        have = st[["PW_WTC", "E_CCDC", "CD2_P"]].notna().all(axis=1) \
            & cl.DeltaClkB_nsps.notna()
        have = have.to_numpy() & (st.end_index >= sp_start).to_numpy()
        n = int(have.sum())
        e_thr = float(thr.loc[sat, "E_CCDC_threshold"])
        c_thr = float(thr.loc[sat, "CD2_P_threshold"])
        g_thr = float(thr.loc[sat, "PW_WTC_threshold"])
        lo, hi = float(cl.LowerBound_nsps.iloc[0]), float(cl.UpperBound_nsps.iloc[0])
        alarms = {
            "PW-WTC": int((have & (st.PW_WTC < g_thr).to_numpy()).sum()),
            "E-CCDC": int((have & (st.E_CCDC > e_thr).to_numpy()).sum()),
            "CD2-P": int((have & (st.CD2_P > c_thr).to_numpy()).sum()),
            "clkBM": int((have & ((cl.DeltaClkB_nsps < lo) | (cl.DeltaClkB_nsps > hi)).to_numpy()).sum()),
        }
        role = "target" if sat == cfg.get("target") else ("control" if cfg.get("target") else "")
        for method, count in alarms.items():
            rows.append({"scene": scene, "satellite": sat, "role": role, "method": method,
                         "sp_positions": n, "sp_alarms": count,
                         "sp_detection_rate": count / n if n else float("nan")})
    table = pd.DataFrame(rows)
    out = ROOT / "results" / mode / scene / "detection_rates.csv"
    table.to_csv(out, index=False)
    print(table.to_string(index=False))


if __name__ == "__main__":
    config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    for scene in (sys.argv[1:] or list(config)):
        run_scene(scene)
