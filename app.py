from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np

# Initialize the Flask app
app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model('/home/seif/.vscode-server/extensions/sourcery.sourcery-1.35.0-linux-x64/DSAI 308/bonus/flask_model_app/best_model.h5')

# Define the columns based on the dataset (features only, no target)
columns = [
    'timestamp', 'hand_temp', 'hand_acc_16g_x',
    'hand_acc_16g_y', 'hand_acc_16g_z', 'hand_acc_6g_x', 'hand_acc_6g_y',
    'hand_acc_6g_z', 'hand_gyro_x', 'hand_gyro_y', 'hand_gyro_z',
    'hand_mag_x', 'hand_mag_y', 'hand_mag_z', 'hand_ori_0', 'hand_ori_1',
    'hand_ori_2', 'hand_ori_3', 'chest_temp', 'chest_acc_16g_x',
    'chest_acc_16g_y', 'chest_acc_16g_z', 'chest_acc_6g_x',
    'chest_acc_6g_y', 'chest_acc_6g_z', 'chest_gyro_x', 'chest_gyro_y',
    'chest_gyro_z', 'chest_mag_x', 'chest_mag_y', 'chest_mag_z',
    'chest_ori_0', 'chest_ori_1', 'chest_ori_2', 'chest_ori_3',
    'ankle_temp', 'ankle_acc_16g_x', 'ankle_acc_16g_y', 'ankle_acc_16g_z',
    'ankle_acc_6g_x', 'ankle_acc_6g_y', 'ankle_acc_6g_z', 'ankle_gyro_x',
    'ankle_gyro_y', 'ankle_gyro_z', 'ankle_mag_x', 'ankle_mag_y',
    'ankle_mag_z', 'ankle_ori_0', 'ankle_ori_1', 'ankle_ori_2',
    'ankle_ori_3'
]

@app.route('/')
def index():
    return render_template('index.html', columns=columns)  # Pass columns to template

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = []
        for col in columns:
            value = request.form.get(col)
            if value is None or value.strip() == "" or value.lower() == "nan":
                return jsonify({'error': f'Missing or invalid value for {col}'}), 400
            input_data.append(float(value))
        input_data = np.array(input_data).reshape(1, -1)
        prediction = model.predict(input_data)
        if np.isnan(prediction[0][0]):
            return jsonify({'error': 'Model returned NaN. Check your input values.'}), 400
        predicted_activity_id = int(np.round(prediction[0][0]))
        return jsonify({'activityID': predicted_activity_id})
    except Exception as e:
        print("Prediction error:", e)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
