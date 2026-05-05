// main.ino
// Hardware : ESP32 + TCA9548A + 5x MPU9250 + SSD1306 OLED (mux bus 7) + NEO-6M GPS
// Comms    : USB Serial -> Raspberry Pi (115200 baud)
// Role     : Sample sensors + GPS -> Kalman filter -> stream CSV rows to Pi
//            Pi handles: windowing, feature extraction, inference, label reply
// Libraries: MPU9250, Adafruit_SSD1306, Adafruit_GFX, TinyGPS++

#include <Wire.h>
#include <MPU9250.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include <TinyGPSPlus.h>

// ── Hardware config ───────────────────────────────────────────────────────────
#define SDA_PIN       21
#define SCL_PIN       22
#define SCREEN_WIDTH  128
#define SCREEN_HEIGHT  64
#define OLED_ADDRESS  0x3C
#define TCA_ADDRESS   0x70
#define OLED_BUS        7

// ── GPS config ────────────────────────────────────────────────────────────────
// NEO-6M connects directly to ESP32 UART2 (not through the I2C multiplexer)
#define GPS_RX_PIN  16   // ESP32 RX2 <- GPS TX
#define GPS_TX_PIN  17   // ESP32 TX2 -> GPS RX (optional)
#define GPS_BAUD    9600

// ── IMU config ────────────────────────────────────────────────────────────────
#define N_IMUS          5
#define N_AXES_PER_IMU 10   // AccX AccY AccZ GyroX GyroY GyroZ MagX MagY MagZ AccMag
#define N_TOTAL_AXES   (N_IMUS * N_AXES_PER_IMU)  // 50 IMU values per row
#define N_GPS_COLS      6   // LAT, LNG, ALT, SPEED, SATS, VALID
#define N_TOTAL_COLS   (N_TOTAL_AXES + N_GPS_COLS) // 56 total per row

// Local axis indices within one IMU
#define AX 0
#define AY 1
#define AZ 2
#define GX 3
#define GY 4
#define GZ 5
#define MX 6
#define MY 7
#define MZ 8
#define AM 9  // acceleration magnitude sqrt(AX^2+AY^2+AZ^2)

#define axIdx(imu, ax) ((imu) * N_AXES_PER_IMU + (ax))

// ── Sampling ──────────────────────────────────────────────────────────────────
#define SAMPLE_PERIOD_US 20000UL   // 20ms = 50Hz

// ── Kalman Filter (1D) ────────────────────────────────────────────────────────
struct KalmanFilter1D {
    float Q;        // process noise  (trust the model)
    float R;        // measurement noise (trust the sensor)
    float P = 1.0f; // estimate error covariance
    float x = 0.0f; // current state estimate

    float update(float z) {
        P += Q;                  // predict: uncertainty grows
        float K = P / (P + R);   // Kalman gain
        x += K * (z - x);        // correct with new measurement
        P *= (1.0f - K);         // update error covariance
        return x;
    }
};

// ── Hardware objects ──────────────────────────────────────────────────────────
Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, -1);
TinyGPSPlus      gps;

MPU9250 mpus[N_IMUS];
const uint8_t IMU_BUS[N_IMUS]    = {  2,        4,           0,           5,           1          };
const char*   IMU_LABELS[N_IMUS] = {"CHEST", "LEFTARM", "RIGHTARM", "LEFTLEG", "RIGHTLEG"};

// ── Kalman filters ────────────────────────────────────────────────────────────
// One per (imu, axis) = 50 total
// Accel [g]:    Q=0.001, R=0.05
// Gyro [deg/s]: Q=0.01,  R=0.3
// Mag [uT]:     Q=0.001, R=0.2
// AccMag [g]:   Q=0.001, R=0.05
KalmanFilter1D kf[N_TOTAL_AXES] = {
    // CHEST
    {0.001f,0.05f},{0.001f,0.05f},{0.001f,0.05f},
    {0.01f, 0.3f }, {0.01f, 0.3f }, {0.01f, 0.3f },
    {0.001f,0.2f }, {0.001f,0.2f }, {0.001f,0.2f },
    {0.001f,0.05f},
    // LEFT ARM
    {0.001f,0.05f},{0.001f,0.05f},{0.001f,0.05f},
    {0.01f, 0.3f }, {0.01f, 0.3f }, {0.01f, 0.3f },
    {0.001f,0.2f }, {0.001f,0.2f }, {0.001f,0.2f },
    {0.001f,0.05f},
    // RIGHT ARM
    {0.001f,0.05f},{0.001f,0.05f},{0.001f,0.05f},
    {0.01f, 0.3f }, {0.01f, 0.3f }, {0.01f, 0.3f },
    {0.001f,0.2f }, {0.001f,0.2f }, {0.001f,0.2f },
    {0.001f,0.05f},
    // LEFT LEG
    {0.001f,0.05f},{0.001f,0.05f},{0.001f,0.05f},
    {0.01f, 0.3f }, {0.01f, 0.3f }, {0.01f, 0.3f },
    {0.001f,0.2f }, {0.001f,0.2f }, {0.001f,0.2f },
    {0.001f,0.05f},
    // RIGHT LEG
    {0.001f,0.05f},{0.001f,0.05f},{0.001f,0.05f},
    {0.01f, 0.3f }, {0.01f, 0.3f }, {0.01f, 0.3f },
    {0.001f,0.2f }, {0.001f,0.2f }, {0.001f,0.2f },
    {0.001f,0.05f},
};

// ── Cached GPS state ──────────────────────────────────────────────────────────
// Updated at 1Hz by the GPS module. Stamped onto every 50Hz IMU row.
// Stays at last known value between GPS updates
float gps_lat   = 0.0f;   // degrees
float gps_lng   = 0.0f;   // degrees
float gps_alt   = 0.0f;   // metres above sea level
float gps_speed = 0.0f;   // km/h
int   gps_sats  = 0;      // number of satellites in view
bool  gps_valid = false;  // false until first satellite fix acquired


// ── State ─────────────────────────────────────────────────────────────────────
String        current_label  = "Starting";
unsigned long last_sample_us = 0;

// ── Helpers ───────────────────────────────────────────────────────────────────
void tcaSelect(uint8_t bus) {
    if (bus > 7) return;
    Wire.beginTransmission(TCA_ADDRESS);
    Wire.write(1 << bus);
    Wire.endTransmission();
}

bool initMPU(int idx) {
    tcaSelect(IMU_BUS[idx]);
    delay(10);
    if (!mpus[idx].setup(0x68)) {
        Serial.print("# FAIL: ");
        Serial.println(IMU_LABELS[idx]);
        return false;
    }

    mpus[idx].calibrateAccelGyro();   // keep sensor still and flat
    mpus[idx].calibrateMag();          // rotate in figure-8

    Serial.print("# OK: ");
    Serial.println(IMU_LABELS[idx]);
    return true;
}

void oledShow(const char* line1, const char* line2 = nullptr) {
    tcaSelect(OLED_BUS);
    display.clearDisplay();
    display.setTextColor(WHITE);
    display.setTextSize(1);
    display.setCursor(0, 0);
    display.println("Activity:");
    display.setTextSize(2);
    display.setCursor(0, 14);
    display.println(line1);
    if (line2) {
        display.setTextSize(1);
        display.setCursor(0, 48);
        display.println(line2);
    }
    display.display();
}

// Read all available GPS bytes and update cached values when a new fix arrives
// Called every loop() — lightweight, non-blocking
void updateGPS() {
    while (Serial2.available()) {
        if (gps.encode(Serial2.read())) {      // true = full NMEA sentence decoded
            if (gps.location.isValid()) {
                gps_lat   = (float)gps.location.lat();
                gps_lng   = (float)gps.location.lng();
                gps_valid = true;
            }
            if (gps.altitude.isValid())
                gps_alt   = (float)gps.altitude.meters();
            if (gps.speed.isValid())
                gps_speed = (float)gps.speed.kmph();
            if (gps.satellites.isValid())
                gps_sats  = (int)gps.satellites.value();
        }
    }
}

// Send CSV header so Pi knows exact column ordering
// All lines starting with '#' are comments — Pi should skip them
void sendHeader() {
    const char* axis_names[N_AXES_PER_IMU] =
        {"AX","AY","AZ","GX","GY","GZ","MX","MY","MZ","AM"};

    Serial.print("# HEADER:");
    for (int i = 0; i < N_IMUS; i++) {
        for (int ax = 0; ax < N_AXES_PER_IMU; ax++) {
            Serial.print(IMU_LABELS[i]);
            Serial.print('_');
            Serial.print(axis_names[ax]);
            Serial.print(',');
        }
    }
    // GPS columns appended after the 50 IMU columns
    Serial.println("GPS_LAT,GPS_LNG,GPS_ALT_M,GPS_SPEED_KMH,GPS_SATS,GPS_VALID");
}

// ─────────────────────────────────────────────────────────────────────────────
void setup() {
    Serial.begin(115200);                              // USB Serial -> Pi
    Serial2.begin(GPS_BAUD, SERIAL_8N1,
                  GPS_RX_PIN, GPS_TX_PIN);             // UART2 -> NEO-6M GPS

    Wire.begin(SDA_PIN, SCL_PIN);
    Wire.setClock(400000);
    delay(200);

// Scan every TCA channel for I2C devices
    for (uint8_t bus = 0; bus < 8; bus++) {
        tcaSelect(bus);
        delay(10);
        for (uint8_t addr = 1; addr < 127; addr++) {
            Wire.beginTransmission(addr);
            if (Wire.endTransmission() == 0) {
                Serial.print("# TCA bus ");
                Serial.print(bus);
                Serial.print(" -> device at 0x");
                Serial.println(addr, HEX);
            }
        }
    }
    // OLED init
    tcaSelect(OLED_BUS);
    delay(100);  // ← give OLED time to power up after TCA select
    if (!display.begin(SSD1306_SWITCHCAPVCC, OLED_ADDRESS)) {
    Serial.println("# OLED FAIL - continuing without display");
}
    Serial.println("# OLED OK");
    display.clearDisplay();
    display.display();
    oledShow("Booting...");


    // IMU init
    int ok_count = 0;
    for (int i = 0; i < N_IMUS; i++) {
        if (initMPU(i)) ok_count++;
        delay(20);
    }

    sendHeader();

    char buf[20];
    snprintf(buf, sizeof(buf), "%d/5 IMUs OK", ok_count);
    oledShow("Ready", buf);

    last_sample_us = micros();
    Serial.println("# READY");
}

// ─────────────────────────────────────────────────────────────────────────────
void loop() {

    // ── 1. Always drain GPS serial buffer (non-blocking, runs every loop) ─────
    // At 9600 baud the GPS sends ~1 NMEA sentence per second.
    // updateGPS() just reads whatever bytes are available right now
    // and updates the cached gps_* variables when a full fix arrives.
    updateGPS();

    // ── 2. 50 Hz IMU sample tick ──────────────────────────────────────────────
    if (micros() - last_sample_us >= SAMPLE_PERIOD_US) {
        last_sample_us += SAMPLE_PERIOD_US;   // add not reset = drift-corrected

        float filtered[N_TOTAL_AXES];

        for (int i = 0; i < N_IMUS; i++) {
            tcaSelect(IMU_BUS[i]);
            if (!mpus[i].update()) continue;

            float raw[N_AXES_PER_IMU];
            raw[AX] = mpus[i].getAccX();       // [g]
            raw[AY] = mpus[i].getAccY();
            raw[AZ] = mpus[i].getAccZ();
            raw[GX] = mpus[i].getGyroX();      // [deg/s]
            raw[GY] = mpus[i].getGyroY();
            raw[GZ] = mpus[i].getGyroZ();
            raw[MX] = mpus[i].getMagX();       // [uT]
            raw[MY] = mpus[i].getMagY();
            raw[MZ] = mpus[i].getMagZ();
            raw[AM] = sqrtf(raw[AX]*raw[AX]
                          + raw[AY]*raw[AY]
                          + raw[AZ]*raw[AZ]); // [g]

            for (int ax = 0; ax < N_AXES_PER_IMU; ax++) {
                filtered[axIdx(i, ax)] = kf[axIdx(i, ax)].update(raw[ax]);
            }
        }

        // ── Send 50 IMU values ────────────────────────────────────────────────
        for (int j = 0; j < N_TOTAL_AXES; j++) {
            Serial.print(filtered[j], 4);
            Serial.print(',');
        }

        // ── Append 6 GPS values (cached from last 1Hz fix) ────────────────────
        // These repeat for ~50 rows between GPS updates
        //GPS_VALID = 0 means no satellite fix yet, Pi should discard GPS cols.
        Serial.print(gps_valid ? gps_lat   : 0.0f, 6); Serial.print(',');
        Serial.print(gps_valid ? gps_lng   : 0.0f, 6); Serial.print(',');
        Serial.print(gps_valid ? gps_alt   : 0.0f, 2); Serial.print(',');
        Serial.print(gps_valid ? gps_speed : 0.0f, 2); Serial.print(',');
        Serial.print(gps_valid ? gps_sats  : 0);        Serial.print(',');
        Serial.print(gps_valid ? 1 : 0);
        Serial.print('\n');
    }

    // ── 3. Receive prediction label from Pi, update OLED ─────────────────────
    // Pi sends plain text after each inference e.g. "Walking\n"
    // Lines starting with '#' are comments and are ignored
    if (Serial.available()) {
        String reply = Serial.readStringUntil('\n');
        reply.trim();
        if (reply.length() > 0 && !reply.startsWith("#")) {
            if (reply == "error") {
                oledShow(current_label.c_str(), "Pi err");
            } else {
                current_label = reply;
                oledShow(current_label.c_str());
            }
        }
    }
}