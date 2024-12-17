import cv2
import face_recognition
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, date
import time
import os

def run_attendance(selected_subject, selected_room):
    print(f"Starting attendance for {selected_subject} in {selected_room}")

    # Check if attendance.csv exists, if not create it with appropriate columns
    if not os.path.exists('attendance.csv'):
        attendance = pd.DataFrame(columns=['Name', 'Subject', 'Room', 'Date', 'Time'])
        attendance.to_csv('attendance.csv', index=False)
    else:
        attendance = pd.read_csv('attendance.csv')

    # Load known face encodings
    with open('face_encodings.pkl', 'rb') as f:
        data = pickle.load(f)
    known_encodings = data['encodings']
    known_names = data['names']

    # Initialize variables
    video_capture = cv2.VideoCapture(0)
    capture_interval = 5  # seconds
    last_capture_time = time.time() - capture_interval

    try:
        while True:
            ret, frame = video_capture.read()
            if not ret:
                break

            current_time = time.time()
            if current_time - last_capture_time >= capture_interval:
                last_capture_time = current_time

                # Resize frame for faster processing
                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

                # Detect faces
                face_locations = face_recognition.face_locations(rgb_small_frame)
                face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

                for face_encoding in face_encodings:
                    matches = face_recognition.compare_faces(known_encodings, face_encoding)
                    face_distances = face_recognition.face_distance(known_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)

                    if matches[best_match_index]:
                        name = known_names[best_match_index]

                        # Mark attendance
                        date_today = date.today().strftime("%Y-%m-%d")
                        time_now = datetime.now().strftime("%H:%M:%S")
                        if not ((attendance['Name'] == name) &
                                (attendance['Subject'] == selected_subject) &
                                (attendance['Room'] == selected_room) &
                                (attendance['Date'] == date_today)).any():
                            new_entry = pd.DataFrame([{
                                'Name': name,
                                'Subject': selected_subject,
                                'Room': selected_room,
                                'Date': date_today,
                                'Time': time_now
                            }])
                            attendance = pd.concat([attendance, new_entry], ignore_index=True)
                            attendance.to_csv('attendance.csv', index=False)
                            print(f"Attendance marked for {name} in {selected_subject} at {selected_room} on {time_now}")
                        else:
                            print(f"{name}'s attendance already marked for {selected_subject} in {selected_room} today.")
                    else:
                        print("Unknown face detected.")

            # Display the resulting frame
            cv2.imshow('Video - Press "q" to exit', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        video_capture.release()
        cv2.destroyAllWindows()
        print(f"Attendance process for {selected_subject} in {selected_room} completed.")

