    
import pandas as pd
import numpy as np
import time
import threading
import RPi.GPIO as GPIO
import Adafruit_DHT
import xgboost as xgb
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from tkinter import *

# ================== MODEL TRAINING ==================
file_path = r"/home/emblab/Desktop/labeled_training_data.csv"
data = pd.read_csv(file_path)

feature_cols = [
    'TEMP_1', 'PPM_1', 'LEVEL_1',
    'TEMP_2', 'PPM_2', 'LEVEL_2',
    'TEMP_3', 'PPM_3', 'LEVEL_3',
    'TEMP_4', 'PPM_4', 'LEVEL_4',
    'TEMP_5', 'PPM_5', 'LEVEL_5'
]
target_cols = ['gas_leak', 'temp_overshoot', 'tank_leak']

# Clean & encode
for col in feature_cols + target_cols:
    data[col] = pd.to_numeric(data[col], errors='coerce')
data.dropna(subset=feature_cols + target_cols, inplace=True)

data['combined_label'] = data[target_cols].astype(str).agg(''.join, axis=1)
label_encoder = LabelEncoder()
data['encoded_label'] = label_encoder.fit_transform(data['combined_label'])

X = data[feature_cols]
y = data['encoded_label']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# Train XGBoost model (clean version without warning)
model = xgb.XGBClassifier(
    objective='multi:softmax',
    num_class=len(label_encoder.classes_),
    eval_metric='mlogloss',   # OK to keep
    random_state=42
)
model.fit(X_train, y_train)

# ================== SENSOR SETUP ==================
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

DHT_PIN = 17
GAS_DO_PIN = 27
ULTRASONIC_TRIG = 23
ULTRASONIC_ECHO = 24

GPIO.setup(GAS_DO_PIN, GPIO.IN)
GPIO.setup(ULTRASONIC_TRIG, GPIO.OUT)
GPIO.setup(ULTRASONIC_ECHO, GPIO.IN)

condition_names = ['gas_leak', 'temp_overshoot', 'tank_leak']

# ================== SENSOR READ FUNCTIONS ==================
def read_dht_sensor():
    humidity, temperature = Adafruit_DHT.read_retry(
        Adafruit_DHT.DHT11, DHT_PIN, retries=2, delay_seconds=0.2
    )
    return temperature, humidity

def read_mq_sensor():
    gas_status = GPIO.input(GAS_DO_PIN)
    return 1 if gas_status == 0 else 0  # 1 = leak detected

def read_ultrasonic_distance():
    GPIO.output(ULTRASONIC_TRIG, True)
    time.sleep(0.00001)
    GPIO.output(ULTRASONIC_TRIG, False)

    start_time = time.time()
    stop_time = time.time()
    timeout = start_time + 0.02  # 20ms timeout

    while GPIO.input(ULTRASONIC_ECHO) == 0 and time.time() < timeout:
        start_time = time.time()
    timeout = time.time() + 0.02
    while GPIO.input(ULTRASONIC_ECHO) == 1 and time.time() < timeout:
        stop_time = time.time()

    distance = (stop_time - start_time) * 34300 / 2
    return round(distance, 2)

# ================== PREDICTION FUNCTIONS ==================
def interpret_prediction(binary_targets):
    return ", ".join([name for name, val in zip(condition_names, binary_targets) if val == 1]) or "No Fault Detected"

def predict_from_sensors():
    temp, humidity = read_dht_sensor()
    gas = read_mq_sensor()
    level = read_ultrasonic_distance()

    if temp is None:
        return None, None, None

    sensor_data = []
    for _ in range(5):
        sensor_data.extend([temp, gas, level])

    new_data = pd.DataFrame([sensor_data], columns=feature_cols)
    new_data_scaled = scaler.transform(new_data)

    preds_encoded = model.predict(new_data_scaled)
    preds_decoded = label_encoder.inverse_transform(preds_encoded)
    preds_binary = [list(map(int, list(label))) for label in preds_decoded]
    return temp, level, preds_binary[0]

# ================== GUI ==================
running = False

def start_monitoring():
    global running
    running = True
    start_btn.config(state=DISABLED)
    stop_btn.config(state=NORMAL)
    threading.Thread(target=update_loop, daemon=True).start()

def stop_monitoring():
    global running
    running = False
    start_btn.config(state=NORMAL)
    stop_btn.config(state=DISABLED)

def update_loop():
    while running:
        temp, level, faults = predict_from_sensors()
        if temp is not None:
            temp_var.set(f"{temp:.1f} °C")
            level_var.set(f"{level:.1f} cm")
            condition_var.set(interpret_prediction(faults))
        time.sleep(0.5)  # faster updates

def on_close():
    stop_monitoring()
    GPIO.cleanup()
    root.destroy()

root = Tk()
root.title("Live Fault Detection Dashboard")
root.geometry("500x350")
root.configure(bg="#eef2f3")

# ----------------- Header Labels -----------------
batch_label = Label(root, text="Batch - 8, Neerav and Shobbiga", 
                    font=("Helvetica", 12, "italic"), bg="#eef2f3", fg="#555555")
batch_label.pack(pady=5)

title_lbl = Label(root, text="Fault Detection System", font=("Helvetica", 16, "bold"), bg="#eef2f3")
title_lbl.pack(pady=5)

frame = Frame(root, bg="white", bd=2, relief="ridge")
frame.pack(padx=20, pady=10, fill="both", expand=True)

temp_var = StringVar()
level_var = StringVar()
condition_var = StringVar()

Label(frame, text="Temperature:", font=("Arial", 12), bg="white").grid(row=0, column=0, sticky=W, padx=10, pady=5)
Label(frame, text="Tank Level:", font=("Arial", 12), bg="white").grid(row=1, column=0, sticky=W, padx=10, pady=5)
Label(frame, text="Predicted Condition:", font=("Arial", 12, "bold"), bg="white").grid(row=2, column=0, sticky=W, padx=10, pady=5)

Label(frame, textvariable=temp_var, font=("Arial", 12), bg="white", fg="#007acc").grid(row=0, column=1, sticky=W)
Label(frame, textvariable=level_var, font=("Arial", 12), bg="white", fg="#007acc").grid(row=1, column=1, sticky=W)
Label(frame, textvariable=condition_var, font=("Arial", 12, "bold"), bg="white", fg="#d62828").grid(row=2, column=1, sticky=W)

btn_frame = Frame(root, bg="#eef2f3")
btn_frame.pack(pady=10)
start_btn = Button(btn_frame, text="Start Monitoring", bg="#28a745", fg="white", width=15, command=start_monitoring)
start_btn.grid(row=0, column=0, padx=5)
stop_btn = Button(btn_frame, text="Stop", bg="#dc3545", fg="white", width=10, state=DISABLED, command=stop_monitoring)
stop_btn.grid(row=0, column=1, padx=5)

root.protocol("WM_DELETE_WINDOW", on_close)
root.mainloop()
