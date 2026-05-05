# Human-Motion-Prediction-Wearable-Hardware-System
Wearable hardware system and the implemented software for data collection (for model training) and activity classification during live deployment.

Wiring Diagram:  

<img width="940" height="488" alt="image" src="https://github.com/user-attachments/assets/5cbb140b-4fa2-4224-bdaa-345878cc154e" />

Components list:

<img width="984" height="714" alt="image" src="https://github.com/user-attachments/assets/1ea74390-b1e7-433e-8c50-e04879268f05" />
<img width="608" height="514" alt="image" src="https://github.com/user-attachments/assets/c42e5b24-544c-4760-b16e-8302395dbed2" />

Assembled Hardware:

<img width="715" height="512" alt="image" src="https://github.com/user-attachments/assets/b0ef11a7-5e0d-4043-82f8-0c808da7d70d" />

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
