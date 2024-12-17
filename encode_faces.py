import face_recognition
import os
import pickle

def encode_faces(dataset_path):
    known_encodings = []
    known_names = []

    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith(('jpg', 'png')):
                img_path = os.path.join(root, file)
                name = os.path.basename(root)
                image = face_recognition.load_image_file(img_path)
                encodings = face_recognition.face_encodings(image)
                if encodings:
                    known_encodings.append(encodings[0])
                    known_names.append(name)
                else:
                    print(f"Face not found in {img_path}")

    data = {'encodings': known_encodings, 'names': known_names}
    with open('face_encodings.pkl', 'wb') as f:
        pickle.dump(data, f)
    print("Encodings saved to face_encodings.pkl")

encode_faces('dataset/')
