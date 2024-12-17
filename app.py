from flask import Flask, render_template, request, jsonify
import threading
import attendance_processor  # This will be the modified main.py

app = Flask(__name__)

# List of subjects and rooms (you can modify these as needed)
subjects = ['Math', 'Science', 'English']
rooms = ['Room 101', 'Room 102', 'Room 103']

@app.route('/')
def teacher_portal():
    return render_template('teacher.html', subjects=subjects, rooms=rooms)

@app.route('/take_attendance', methods=['POST'])
def take_attendance():
    subject = request.form.get('subject')
    room = request.form.get('room')

    if not subject or not room:
        return jsonify({'status': 'error', 'message': 'Subject and Room are required.'}), 400

    # Run the attendance process in a separate thread to avoid blocking
    threading.Thread(target=attendance_processor.run_attendance, args=(subject, room)).start()

    return jsonify({'status': 'success', 'message': f'Attendance process started for {subject} in {room}.'})

if __name__ == '__main__':
    app.run(debug=True)
