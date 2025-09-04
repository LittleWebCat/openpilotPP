#!/usr/bin/env python3

# --------------------------------------------------------------------------- #
#  Imports
# --------------------------------------------------------------------------- #
import argparse, csv, sys
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import cereal.messaging as messaging

try:                                   # optional for --preload
    from tools.lib.logreader import LogReader
except ImportError:
    LogReader = None


# --------------------------------------------------------------------------- #
#  Args
# --------------------------------------------------------------------------- #

def build_parser():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--hz",       type=float, default=100, help="animation refresh rate (Hz)")
    p.add_argument("--preload",  type=Path, help="rlog / *.bz2 file to draw before going live")
    p.add_argument("--out",      type=Path, help="CSV file for storing fixes")
    p.add_argument("--name",     default="kalman_track_CS", help="stem for auto‑named CSV")
    return p

args = build_parser().parse_args()


# --------------------------------------------------------------------------- #
#  Output CSV setup
# --------------------------------------------------------------------------- #
if args.out:
    csv_path = args.out.expanduser()
else:
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    csv_path = Path(f"{args.name}_{stamp}.csv")

csv_path.parent.mkdir(parents=True, exist_ok=True)
csv_file   = csv_path.open("w", newline="", buffering=1)     # line‑buffered
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["unix_ms", "latitude", "longitude",
                     "altitude_m", "speed_mps", "bearing_deg",
                     "desired_curvature_radpm"])
print(f"Logging Kalman fixes ➜ {csv_path.resolve()}")


# --------------------------------------------------------------------------- #
#  Figure 2 – live curvature signals
# --------------------------------------------------------------------------- #
fig_cv, ax_cv = plt.subplots(figsize=(7, 3))
fig_cv.canvas.manager.set_window_title("Curvature signals (live)")

line_cal,   = ax_cv.plot([], [], label="cal_cv (acc/ v²)")
line_cc,    = ax_cv.plot([], [], label="cc_cv (actuator)")
line_model, = ax_cv.plot([], [], label="model_cv (modelV2)")

ax_sa = ax_cv.twiny()

line_cc_SAD, = ax_sa.plot([], [], color="tab:orange",
                          label="cc_SAD (actuator)")
line_CS_SAD, = ax_sa.plot([], [], color="tab:green",
                          label="CS_SAD (carState)")

ax_cv.set_xlabel("sample #")
ax_cv.set_ylabel("curvature  [rad m⁻¹]")
# ax_sa.set_ylabel("steering angle [deg]")
ax_cv.grid(True)

# single legend combining both axes
lines_all = [line_cal, line_cc, line_model, line_cc_SAD, line_CS_SAD]
ax_cv.legend(lines_all, [l.get_label() for l in lines_all], loc="upper left")

# rolling history (trimmed to last N points to keep draw speed high)
N_HISTORY = 3_000
xs = []            # simple sample counter
hist_cal   = []
hist_cc    = []
hist_model = []
hist_cc_SAD  = []
hist_CS_SAD  = []



# --------------------------------------------------------------------------- #
#  Historical preload
# --------------------------------------------------------------------------- #
lats, lons = [], []

if args.preload and args.preload.exists():
    if LogReader is None:
        print("Preload requested but tools.lib.logreader not available; skipping.")
    else:
        print(f"Pre‑loading {args.preload} …")
        for ev in LogReader(str(args.preload)):
            if ev.which() != "liveLocationKalmanDEPRECATED":
                continue
            g = ev.liveLocationKalmanDEPRECATED
            if g.status != "valid" or not g.positionGeodetic.valid:
                continue
            lat, lon, *_ = g.positionGeodetic.value
            lats.append(lat);  lons.append(lon)
        print(f"  Loaded {len(lats)} points.")


# --------------------------------------------------------------------------- #
#  Projection helper (simple equirectangular)
# --------------------------------------------------------------------------- #

def equirect_xy(lat, lon, lat0, lon0):
    R = 6_378_137.0
    dlat = np.deg2rad(lat - lat0)
    dlon = np.deg2rad(lon - lon0)
    x = R * np.cos(np.deg2rad(lat0)) * dlon
    y = R * dlat
    return x, y


# --------------------------------------------------------------------------- #
#  Live subscription
# --------------------------------------------------------------------------- #
sm = messaging.SubMaster(["liveLocationKalmanDEPRECATED", "controlsState","modelV2","carControl","livePose","carState"])
origin_latlon = [None, None]           # will hold first lat/lon

desired_curvature_latest = np.nan      # updated whenever controlsState arrives


# --------------------------------------------------------------------------- #
#  Matplotlib boiler‑plate
# --------------------------------------------------------------------------- #
fig, ax = plt.subplots(figsize=(6, 6))
fig.canvas.manager.set_window_title("Kalman GPS track (live)")

track_line, = ax.plot([], [], lw=2, color="tab:blue", label="path history")
curr_pt,   = ax.plot([], [], "ro", ms=6, label="current fix")

ax.set_xlabel("X [m]"); ax.set_ylabel("Y [m]")
ax.set_aspect("equal", adjustable="box")
ax.grid(True); ax.legend(loc="upper left")


# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #

def horiz_speed_and_bearing(v_ned):
    """Return horizontal speed (m/s) and bearing deg (0°=N, 90°=E)"""
    vn, ve, *_ = v_ned
    speed = float(np.hypot(vn, ve))
    bearing_deg = (np.degrees(np.arctan2(ve, vn)) + 360.) % 360.
    return speed, bearing_deg


# --------------------------------------------------------------------------- #
#  Animation callback
# --------------------------------------------------------------------------- #
lp_acc_y = 0.0
CS_vEgo = 0.0

def update(_):
    global desired_curvature_latest, lp_acc_y, CS_vEgo, model_cv, cc_cv,cc_SAD, CS_SAD

    sm.update(0)                       # non‑blocking poll


    if sm.updated['modelV2']:
        model_v2 = sm['modelV2']
        model_cv = model_v2.action.desiredCurvature
        # print(f'model_cv:{model_cv}')

    if sm.updated['carControl']:
        cc = sm['carControl']
        cc_cv = cc.actuators.curvature
        cc_SAD = cc.actuators.steeringAngleDeg
        print(f'cc_cv:{cc_SAD}')

    if sm.updated['livePose']:
        lp = sm['livePose']
        lp_acc_x = lp.accelerationDevice.x
        lp_acc_y = lp.accelerationDevice.y
        # print(f'lp_acc_x:{lp_acc_x}')

    if sm.updated['carState']:
        CS = sm['carState']
        CS_vEgo = CS.vEgo
        CS_SAD = CS.steeringAngleDeg
        # print(f'CS_vEgo:{CS_vEgo}')
    if CS_vEgo > 0.01:
        cal_cv = lp_acc_y/ CS_vEgo**2
    else:
        cal_cv = 0
    # print(f'cal_cv:{cal_cv}')

    # --- pull latest desired curvature if present ---------------------------
    if sm.updated["controlsState"]:
        cs = sm["controlsState"]
        # Field name changed across releases; try both
        if hasattr(cs, "curvatureDesired"):
            desired_curvature_latest = float(cs.curvatureDesired)
        elif hasattr(cs, "desiredCurvature"):
            desired_curvature_latest = float(cs.desiredCurvature)
        else:
            desired_curvature_latest = np.nan

    # --- handle Kalman fix ---------------------------------------------------
    if sm.updated["liveLocationKalmanDEPRECATED"]:
        g = sm["liveLocationKalmanDEPRECATED"]

        # validity guards
        if g.status != "valid" or not g.positionGeodetic.valid:
            return track_line, curr_pt

        lat, lon, alt = g.positionGeodetic.value
        if origin_latlon[0] is None:            # 1st fix becomes origin
            origin_latlon[:] = [lat, lon]

        lats.append(lat); lons.append(lon)

        # derive speed & bearing
        if g.velocityNED.valid:
            speed, bearing = horiz_speed_and_bearing(g.velocityNED.value)
        else:
            speed   = np.nan
            bearing = np.nan

        # write to CSV
        csv_writer.writerow([g.unixTimestampMillis,
                             lat, lon, alt,
                             speed, bearing,
                             desired_curvature_latest])
        if len(lats) % 100 == 0:
            print(f"  stored {len(lats)} fixes…")

        # update plot
        x, y = equirect_xy(np.array(lats), np.array(lons),
                           origin_latlon[0], origin_latlon[1])

        track_line.set_data(x, y)
        curr_pt.set_data(x[-1:], y[-1:])

        curv_txt = (f"{desired_curvature_latest:.4f}" if not np.isnan(desired_curvature_latest)
                     else "n/a")
        ax.relim(); ax.autoscale_view()
        ax.set_title(f"Fixes: {len(lats)}  Lat: {lat:.6f}  Lon: {lon:.6f}  Curv: {curv_txt}")



        if not np.isnan(cal_cv) or not np.isnan(cc_cv) or not np.isnan(model_cv):
            xs.append(len(xs))
            hist_cal.append(cal_cv)
            hist_cc.append(cc_cv)
            hist_model.append(model_cv)
            hist_cc_SAD.append(cc_SAD)
            hist_CS_SAD.append(CS_SAD)

            # trim history to last N points
            if len(xs) > N_HISTORY:
                xs[:len(xs)-N_HISTORY]             = []
                hist_cal[:len(hist_cal)-N_HISTORY] = []
                hist_cc [:len(hist_cc) -N_HISTORY] = []
                hist_model[:len(hist_model)-N_HISTORY] = []
                hist_cc_SAD [:len(hist_cc_SAD) -N_HISTORY] = []
                hist_CS_SAD [:len(hist_CS_SAD) -N_HISTORY] = []
            # update Plot 2
            # line_cal.set_data(xs, hist_cal)
            line_cc.set_data(xs, hist_cc)
            line_model.set_data(xs, hist_model)
            line_cc_SAD.set_data(xs, hist_cc_SAD)
            line_CS_SAD.set_data(xs, hist_CS_SAD)
            ax_cv.relim(); ax_cv.autoscale_view()
            # ax_sa.relim(); ax_sa.autoscale_view()

    return track_line, curr_pt,line_cal, line_cc, line_model, line_CS_SAD, line_cc_SAD


# --------------------------------------------------------------------------- #
#  Kick‑off
# --------------------------------------------------------------------------- #
ani = FuncAnimation(fig,
                    update,
                    interval=int(1000 / args.hz),
                    cache_frame_data=False)
ani_cv = FuncAnimation(fig_cv,
                       update,
                       interval=int(1000 / max(args.hz, 1e-3)),
                       cache_frame_data=False)
try:
    plt.show()
finally:
    csv_file.close()
    print(f"CSV closed with {len(lats)} total fixes.")
