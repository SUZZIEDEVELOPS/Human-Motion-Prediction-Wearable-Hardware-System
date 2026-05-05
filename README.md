# Human-Motion-Prediction-Wearable-Hardware-System
Wearable hardware system and the implemented software for data collection (for model training) and activity classification during live deployment.

Wiring Diagram:  

<img width="940" height="488" alt="image" src="https://github.com/user-attachments/assets/5cbb140b-4fa2-4224-bdaa-345878cc154e" />

Components list:

<img width="1788" height="1295" alt="image" src="https://github.com/user-attachments/assets/420c1d94-9046-4c89-ae1d-08dba36eb9fb" />

<img width="724" height="609" alt="image" src="https://github.com/user-attachments/assets/a0a1a9fa-6824-47c9-8994-d452fd64f113" />


Assembled Hardware:

<img width="1012" height="759" alt="Untitled design (4)" src="https://github.com/user-attachments/assets/c78c6bac-7d62-48fc-bd90-dfb0cd99de4a" />


Software- Summary of the systems script types:

1.	Training scripts
collect_data.py
-	Run on the Pi and records raw sensor stream to CSV.
extract_features.py 
-	Run on laptop and turns the raw CSV into feature CSV.
main.py
-	Loads hardware CSV data, windows it, extracts features, selects features, normalises, trains AdaBoost, saves three files:

models/activity_model.pk1 → trained AdaBoost model
models/scaler.pk1 → fitted StandardScaler
models/selected_features.json → list of 150 selected feature names

2.	ESP32 Firmware to be flashed onto the ESP32.
main.ino_esp32
-	Reads sensors, Kalman filters, streams CSV to Pi, receives label back, shows on OLED. 
-	Runs continuously on the ESP32.

3.	PI inference script to run on the Raspberry Pi.
pi_realtime.py
-	Reads ESP32 stream, buffers, windows, extracts features, selects, normalises, runs 
-	AdaBoost, majority vote prediction stabilisation, sends label back. Runs continuously on the Pi.


Full documentation explaining the process of building the hardware and the software implementation can be found in the Documentation_HMP pdf file.
