from flask import Flask, jsonify, render_template
from flask_cors import CORS
import pandas as pd

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route('/attendance')
def get_attendance():
    data = pd.read_csv('attendance.csv')
    return jsonify(data.to_dict(orient='records'))

@app.route('/')
def serve_index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
