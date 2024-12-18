import cv2
import face_recognition
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, date
import time
import os
import dlib

# Set this to True to use IP Camera, False to use webcam
USE_IP_CAMERA =True

def call(som, rom):
    # Check if attendance.csv exists, if not create it
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

    # Choose video source
    if USE_IP_CAMERA:
        ip_camera_url = "http://172.20.10.2:8080/video"
        video_capture = cv2.VideoCapture(ip_camera_url)
        if not video_capture.isOpened():
            print(f"Could not open IP camera stream at {ip_camera_url}.")
            return
    else:
        video_capture = cv2.VideoCapture(0)
        if not video_capture.isOpened():
            print("Could not open webcam (index 0). Please ensure a webcam is connected.")
            return

    # Load facial landmark predictor
    predictor_path = r"C:\Users\nihal\Desktop\Test 6  - Nihal\attendance_env\Lib\site-packages\cv2\data\shape_predictor_68_face_landmarks.dat"
    if not os.path.exists(predictor_path):
        raise FileNotFoundError("shape_predictor_68_face_landmarks.dat not found.")

    predictor = dlib.shape_predictor(predictor_path)
    detector = dlib.get_frontal_face_detector()

    capture_interval = 5  # seconds
    last_capture_time = time.time() - capture_interval

    start_time = time.time()

    def get_zoom_factor(elapsed):
        # Zoom schedule
        if elapsed < 3:
            return 1.0
        elif elapsed < 6:
            return 1.3
        elif elapsed < 9:
            return 1.6
        elif elapsed < 12:
            return 2.5
        else:
            return 1.0

    cv2.namedWindow('Video - Press "q" to exit', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Video - Press "q" to exit', 640, 480)
    cv2.namedWindow('Aligned Face', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Aligned Face', 320, 240)

    aligned_face_shown = False

    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("Failed to retrieve frame from camera source.")
            break

        current_time = time.time()
        elapsed = current_time - start_time

        zoom_factor = get_zoom_factor(elapsed)

        h, w = frame.shape[:2]
        new_w = int(w / zoom_factor)
        new_h = int(h / zoom_factor)
        x1 = (w - new_w) // 2
        y1 = (h - new_h) // 2
        zoomed_frame = frame[y1:y1+new_h, x1:x1+new_w]
        frame = cv2.resize(zoomed_frame, (w, h), interpolation=cv2.INTER_LINEAR)

        face_locations = []
        name = ""  # To store the name of the detected person

        if current_time - last_capture_time >= capture_interval:
            last_capture_time = current_time

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                face_distances = face_recognition.face_distance(known_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)

                # stricter threshold
                if face_distances[best_match_index] < 0.5:
                    name = known_names[best_match_index]
                    date_today = date.today().strftime("%Y-%m-%d")
                    time_now = datetime.now().strftime("%H:%M:%S")
                    if not ((attendance['Name'] == name) &
                            (attendance['Subject'] == str(som)) &
                            (attendance['Room'] == str(rom)) &
                            (attendance['Date'] == date_today)).any():
                        new_entry = pd.DataFrame([{
                            'Name': name,
                            'Subject': str(som),
                            'Room': str(rom),
                            'Date': date_today,
                            'Time': time_now
                        }])
                        attendance = pd.concat([attendance, new_entry], ignore_index=True)
                        attendance.to_csv('attendance.csv', index=False)
                        print(f"Attendance marked for {name} in {str(som)} at {str(rom)} on {time_now}")
                    else:
                        print(f"{name}'s attendance already marked for {str(som)} in {str(rom)} today.")
                else:
                    name = "Unknown"
                    print("Unknown face detected.")

            # Process aligned faces after detection
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            detections = detector(gray_frame)
            aligned_face_shown = False
            for det in detections:
                landmarks = predictor(gray_frame, det)
                aligned_face = align_face(frame, landmarks)
                cv2.imshow('Aligned Face', aligned_face)
                aligned_face_shown = True

        # Draw bounding boxes and names
        for (top, right, bottom, left) in face_locations:
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

        # Show the main video frame
        cv2.imshow('Video - Press "q" to exit', frame)

        # Check for 'q' after showing all windows
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()


def align_face(image, landmarks):
    points = np.array([(landmarks.part(n).x, landmarks.part(n).y) for n in range(68)])
    left_eye = np.mean(points[36:42], axis=0)
    right_eye = np.mean(points[42:48], axis=0)
    eye_center = ((left_eye[0] + right_eye[0]) / 2, (left_eye[1] + right_eye[1]) / 2)
    angle = np.arctan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0]) * 180 / np.pi
    h, w = image.shape[:2]
    rot_mat = cv2.getRotationMatrix2D(eye_center, angle, 1)
    aligned = cv2.warpAffine(image, rot_mat, (w, h), flags=cv2.INTER_CUBIC)
    return aligned


if __name__ == "__main__":
    call('Mathematics', 'Room 101')
