#import re #regular expressions module so you can work with pattern matching in strings.

# ---------- FILENAME PARSING ----------
#def parse_filename(filename):
 #   """
  #  Handles filenames like:
   # - Prelim_Cycling_stationary_Pocket.csv
    #- Prelim_RunningUphill_UpperArm.csv
    #- SittingPrelim1_pocket_Acceleration.csv
    #"""

    #name = filename.replace(".csv", "")
    #parts = name.split("_")

    # Case 1: starts with Prelim_
    #if parts[0].lower() == "prelim":
      #  mount = parts[-1].lower()
       # activity = "_".join(parts[1:-1]).lower()

    # Case 2: activity first (legacy format)
    #else:
     #   match = re.match(r"([A-Za-z]+?)(?=Prelim|\d|_)", parts[0])
      #  if not match:
       #     raise ValueError(f"Cannot parse activity from {filename}")

        #activity = match.group(1).lower()
        #mount = parts[1].lower() if len(parts) > 1 else None

import os, re
import pandas as pd

def load_csv(path):
    return pd.read_csv(path)

def parse_filename(filename: str):
    name = os.path.splitext(filename)[0]

    # split on underscores and parentheses
    tokens = [t for t in re.split(r"[_()]", name) if t]

    # normalize: lowercase, strip digits, remove 'prelim' anywhere
    norm = []
    for t in tokens:
        t = t.lower()
        t = re.sub(r"\d+", "", t)
        t = t.replace("prelim", "")
        t = t.strip()
        if t:
            norm.append(t)

    # ---- mount detection (FIXED) ----
    mount = None
    if "pocket" in norm:
        mount = "pocket"
    elif "upperarm" in norm or ("upper" in norm and "arm" in norm):
        mount = "upperarm"

    # remove mount tokens from activity candidates
    activity_tokens = [t for t in norm if t not in {"pocket", "upperarm", "upper", "arm"}]

    # also remove sensor-ish tokens if they appear in filenames
    sensor_words = {
        "acceleration", "accelerometer",
        "gyro", "gyroscope", "angular", "angularvelocity", "angular_velocity",
        "magneticfield", "magnetometer",
        "orientation", "position"
    }
    activity_tokens = [t for t in activity_tokens if t not in sensor_words]

    # ---- activity map (ALL LOWERCASE KEYS) ----
    activity_map = {
        "walkingdownstairs": "walkingdownstairs",
        "walkinguphill": "walkinguphill",
        "walkingupstairs": "walkingupstairs",
        "runninguphill": "runninguphill",
        "cyclingstationary": "cyclingstationary",
        "cycling": "cycling",
        "walking": "walking",
        "running": "running",
        "sitting": "sitting",
        "standing": "standing",
        "rowing": "rowing",
        "jumping": "jumping",
        "layingdown": "layingdown",
    }

    tokens = activity_tokens  # from earlier cleaning

    # 1️⃣ Single-token match
    for tok in tokens:
        if tok in activity_map:
            return activity_map[tok], mount

    # 2️⃣ Pairwise concatenation
    for i in range(len(tokens) - 1):
        pair = tokens[i] + tokens[i + 1]
        if pair in activity_map:
            return activity_map[pair], mount

    # 3️⃣ Full concatenation
    joined_all = "".join(tokens)
    if joined_all in activity_map:
        return activity_map[joined_all], mount

    return "unknown", mount


# ---------- SENSOR DETECTION ----------
def detect_sensors(df):
    sensors = set()
    cols = df.columns.str.lower()

    if any("acceleration" in c for c in cols):
        sensors.add("acceleration")

    if any(("angular" in c) or ("gyro" in c) for c in cols):
        sensors.add("angular_velocity")

    if any("magnetic" in c for c in cols):
        sensors.add("magnetic_field")

    if any("orientation" in c for c in cols):
        sensors.add("orientation")

    if any(c in ["latitude", "longitude", "altitude", "speed", "course", "hacc"] for c in cols):
        sensors.add("position")

    return list(sensors)


# ---------- FULL FILE PROCESS ----------
def process_file(path):
    df = load_csv(path)

    filename = os.path.basename(path)
    activity, mount = parse_filename(filename)

    sensors_present = detect_sensors(df)

    return df, activity, mount, sensors_present


