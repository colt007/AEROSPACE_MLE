import requests
import pandas as pd
from pathlib import Path

# 1. Point to the RAW CMAPSS data (not the Parquet)
# Adjust this path if your raw text file is located somewhere else
raw_file_path = Path("C:/projects_data/data/raw/train_FD001.txt")

print("Loading raw engine data...")
# CMAPSS text files are space-separated with no headers
columns = ['unit_nr', 'time_cycles', 'op_setting_1', 'op_setting_2', 'op_setting_3'] + [f'sensor_{i}' for i in range(1, 22)]
df = pd.read_csv(raw_file_path, sep=r'\s+', header=None, names=columns)

# 2. Extract Engine No. 1
engine_1 = df[df['unit_nr'] == 1].copy()

# 3. Grab the LAST 30 flights before catastrophic failure
# If the model understands physics, it should predict an RUL close to 0
window_df = engine_1.tail(30).copy()

# 4. Drop 'unit_nr' and 'time_cycles' to match the 24-column API schema
window_df = window_df.drop(columns=['unit_nr', 'time_cycles'])

# Convert the Pandas DataFrame into a native Python list of lists
real_window = window_df.values.tolist()

payload = {
    "window_data": real_window
}

# 5. Fire the payload at the local server
print(f"Sending 30 flights of physical engine data to the API...")
response = requests.post("http://127.0.0.1:8000/predict", json=payload)

print(f"Status Code: {response.status_code}")
try:
    print(f"Predicted RUL: {response.json()['predicted_rul']} cycles remaining")
except Exception as e:
    print(f"Error reading response: {response.text}")