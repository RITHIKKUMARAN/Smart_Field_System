from flask import Flask, request, jsonify
import json
import os
from datetime import datetime

app = Flask(__name__)

# File to store sensor data
DATA_FILE = "sensor_data.json"

# Initialize the data file if it doesn't exist
if not os.path.exists(DATA_FILE):
    with open(DATA_FILE, 'w') as f:
        json.dump([], f)

@app.route('/data', methods=['POST'])
def receive_data():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400

        # Add a proper timestamp
        data["timestamp"] = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

        # Load existing data
        with open(DATA_FILE, 'r') as f:
            existing_data = json.load(f)

        # Append new data
        existing_data.append(data)

        # Keep only the last 100 entries to manage file size
        existing_data = existing_data[-100:]

        # Save updated data
        with open(DATA_FILE, 'w') as f:
            json.dump(existing_data, f, indent=2)

        return jsonify({"status": "success"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8002)
