import serial
import csv
import sys
import time
import os

# ── Config ────────────────────────────────────────────────────────────────────
SERIAL_PORT = '/dev/ttyUSB0'
BAUD_RATE   = 115200
OUTPUT_DIR  = "data_hardware_raw"

# ── Column names ──────────────────────────────────────────────────────────────
IMU_NAMES  = ["CHEST", "LEFTARM", "RIGHTARM", "LEFTLEG", "RIGHTLEG"]
AXIS_NAMES = ["AccX", "AccY", "AccZ", "GyroX", "GyroY", "GyroZ",
              "MagX", "MagY", "MagZ", "AccMag"]
IMU_COLS   = [f"{imu}_{ax}" for imu in IMU_NAMES for ax in AXIS_NAMES]
GPS_COLS   = ["GPS_LAT", "GPS_LNG", "GPS_ALT_M",
              "GPS_SPEED_KMH", "GPS_SATS", "GPS_VALID"]
ALL_COLS   = IMU_COLS + GPS_COLS


def collect(activity, duration_sec):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    existing = [f for f in os.listdir(OUTPUT_DIR)
                if f.startswith(activity + "_") and f.endswith(".csv")]
    session_num = len(existing) + 1
    filename = f"{OUTPUT_DIR}/{activity}_{session_num}.csv"

    print(f"Opening {SERIAL_PORT}...")
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=2)
    except Exception as e:
        print(f"Could not open serial port: {e}")
        print("Try: ls /dev/tty* to find the correct port")
        sys.exit(1)

    print(f"Connected.")
    print(f"Activity : {activity}")
    print(f"Duration : {duration_sec} seconds")
    print(f"Saving to: {filename}")
    print()
    print("Starting in 3 seconds — get into position...")
    time.sleep(3)
    print(">>> RECORDING NOW <<<")

    start_time = time.time()
    rows_saved = 0

    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(ALL_COLS)

        while time.time() - start_time < duration_sec:
            try:
                raw = ser.readline().decode('utf-8', errors='ignore').strip()
            except Exception:
                continue

            if not raw or raw.startswith('#'):
                continue

            values = raw.split(',')

            if len(values) != 56:
                continue

            writer.writerow(values)
            rows_saved += 1

            elapsed   = time.time() - start_time
            remaining = int(duration_sec - elapsed)
            if rows_saved % 50 == 0:
                print(f"  {int(elapsed)}s elapsed | {remaining}s remaining "
                      f"| {rows_saved} rows saved")

    ser.close()
    print()
    print(f"DONE — {rows_saved} rows saved to {filename}")
    print(f"Expected ~{duration_sec * 50} rows at 50Hz")
    return filename


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python3 collect_data.py <activity> <duration_seconds>")
        print()
        print("Examples:")
        print("  python3 collect_data.py walking  60")
        print("  python3 collect_data.py running  60")
        print("  python3 collect_data.py sitting  60")
        print("  python3 collect_data.py standing 60")
        sys.exit(1)

    activity     = sys.argv[1].lower().strip()
    duration_sec = int(sys.argv[2])

    collect(activity, duration_sec)