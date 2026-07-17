"""AOI orchestrator: draw-a-box -> full PS-6 deliverables, end to end.

  python aoi_run.py --bbox W S E N [--year 2021] [--name myaoi] [--resume]

Runs, in a workspace under aoi_runs/<name>/:
  1. aoi_prepare      GEE fetch (training-recipe composites) + pseudo-fields
  2. aoi_classify     8-class LGB crop map (label-free)
  3. weather_et0      NASA POWER grid snapped to the AOI + FAO-56 ET0
  4. stress_indices   per-field NDVI/NDRE/NDMI/SSM/VCI series
  5. sowing_detect    per-field sowing from NDVI green-up
  6. water_balance    FAO-56 buckets per (cell x crop x sow-bin), 8-day deficit
  7. advisory         5-level per-field advisory
  8. stress_lstm --infer   satellite-only Ks emulator (UP-trained)
  9. advisory_maps    advisory PNGs + GeoTIFFs
 10. deficit_maps     8-day deficit PNGs + multi-band GeoTIFFs
 11. dashboard_data   dashboard.html (self-contained)

Every stage is the SAME script the UP demonstrator uses — the AOI is
retargeted purely through AGRI_* environment variables (see stress_common).

Scientific gate: the classifier was trained on UP/Bihar/Odisha/Rajasthan
2021-22; outside a northern-India box the crop map is unvalidated, so bboxes
beyond it are refused (override with --experimental at your own risk).
"""
import argparse
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
PIPE = os.path.join(ROOT, "pipeline")            # stage scripts live here
# Gate: Indo-Gangetic plain + the four pilot regions (W, S, E, N). South of
# ~19N the model has zero training support (different crops/calendars).
SUPPORTED = (68.0, 19.0, 89.0, 32.5)
PILOTS = {  # validated pilot bboxes for the "validated vs experimental" badge
    "UP": (81.13, 27.07, 82.74, 28.33), "BIHAR": (87.20, 25.27, 88.05, 25.88),
    "ODISHA": (83.00, 19.01, 83.97, 19.92), "RAJASTHAN": (76.25, 24.41, 77.31, 25.43),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bbox", nargs=4, type=float, required=True,
                    metavar=("W", "S", "E", "N"))
    ap.add_argument("--year", type=int, default=2021,
                    help="agricultural year Y (season = Jun Y .. Apr Y+1)")
    ap.add_argument("--name", default=None)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--resume", action="store_true",
                    help="skip stages whose outputs already exist")
    ap.add_argument("--experimental", action="store_true",
                    help="allow AOIs outside the supported northern-India zone")
    args = ap.parse_args()
    w, s, e, n = args.bbox

    if not (SUPPORTED[0] <= w and e <= SUPPORTED[2]
            and SUPPORTED[1] <= s and n <= SUPPORTED[3]):
        if not args.experimental:
            raise SystemExit(
                f"AOI outside the supported zone {SUPPORTED} (northern India; "
                "the classifier has no training data further south). "
                "Pass --experimental to run anyway — crop labels there are "
                "unvalidated.")
        print("WARNING: outside supported zone — crop map is unvalidated here.")
    validated = any(bw <= w and e <= be and bs <= s and n <= bn
                    for bw, bs, be, bn in PILOTS.values())
    print(f"AOI [{w},{s},{e},{n}] year {args.year} — "
          + ("VALIDATED pilot region" if validated else
               "supported (experimental accuracy: outside the labeled pilots)"))

    name = args.name or f"{w:.2f}_{s:.2f}_{e:.2f}_{n:.2f}_{args.year}"
    ws = os.path.join(ROOT, "aoi_runs", name)
    out = os.path.join(ws, "moisture_stress")
    os.makedirs(out, exist_ok=True)

    env = dict(os.environ,
               AGRI_DATA_DIR=ws,
               AGRI_S2_DIR=os.path.join(ws, "s2raw"),
               AGRI_S2RE_DIR=os.path.join(ws, "s2reraw"),
               AGRI_SEASON_DIR=os.path.join(ws, "season"),
               AGRI_OUT_DIR=out,
               AGRI_CHIP_PREFIXES="AOI_",
               AGRI_SEASON_YEAR=str(args.year),
               AGRI_WEATHER_BBOX=f"{w},{s},{e},{n}",
               AGRI_REGION_NAME=f"AOI {name}"
               + ("" if validated else " (experimental)"))

    def done(*paths):
        return all(os.path.exists(os.path.join(ws, p)) for p in paths)

    stages = [
        ("prepare", [sys.executable, os.path.join(PIPE, "aoi_prepare.py"), "--bbox", str(w), str(s),
                     str(e), str(n), "--year", str(args.year), "--workspace", ws,
                     "--workers", str(args.workers)],
         lambda: done("aoi_meta.json")),
        ("classify", [sys.executable, os.path.join(PIPE, "aoi_classify.py")],
         lambda: done("moisture_stress/crop_map.npz")),
        ("weather", [sys.executable, os.path.join(PIPE, "weather_et0.py")],
         lambda: done("moisture_stress/weather_daily.csv")),
        ("indices", [sys.executable, os.path.join(PIPE, "stress_indices.py")],
         lambda: done("moisture_stress/field_timeseries.npz")),
        ("sowing", [sys.executable, os.path.join(PIPE, "sowing_detect.py")],
         lambda: done("moisture_stress/sowing.npz")),
        ("water_balance", [sys.executable, os.path.join(PIPE, "water_balance.py")],
         lambda: done("moisture_stress/water_balance.json")),
        ("advisory", [sys.executable, os.path.join(PIPE, "advisory.py")],
         lambda: done("moisture_stress/advisory.npz")),
        ("lstm", [sys.executable, os.path.join(PIPE, "stress_lstm.py"), "--infer"],
         lambda: done("moisture_stress/lstm_ks.npz")),
        ("advisory_maps", [sys.executable, os.path.join(PIPE, "advisory_maps.py")],
         lambda: done("moisture_stress/maps/region_grid.png")),
        ("deficit_maps", [sys.executable, os.path.join(PIPE, "deficit_maps.py")],
         lambda: done("moisture_stress/maps/deficit_grid.png")),
        ("dashboard", [sys.executable, os.path.join(PIPE, "dashboard_data.py")],
         lambda: done("moisture_stress/dashboard.html")),
    ]

    t_all = time.time()
    for label, cmd, is_done in stages:
        if args.resume and is_done():
            print(f"== {label}: already done, skipping")
            continue
        t0 = time.time()
        print(f"== {label} ==")
        r = subprocess.run(cmd, cwd=ROOT, env=env)
        if r.returncode != 0:
            raise SystemExit(f"stage '{label}' failed (exit {r.returncode})")
        print(f"== {label} done in {time.time()-t0:.0f}s")

    with open(os.path.join(ws, "run_info.json"), "w") as f:
        json.dump(dict(bbox=[w, s, e, n], year=args.year, validated=validated,
                       total_s=round(time.time() - t_all, 1)), f, indent=1)
    print(f"\nALL DONE in {time.time()-t_all:.0f}s -> {out}\\dashboard.html")


if __name__ == "__main__":
    main()
