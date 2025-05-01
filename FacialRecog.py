import cv2
import os
import numpy as np

# Configurations
data_dir = "stored_faces"
trainer_file = "trainer.yml"
threshold = 100  # Increase for looser match; lower for stricter

# Ensure face storage directory exists
os.makedirs(data_dir, exist_ok=True)

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Initialize face recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

def capture_face():
    """Capture 5 face images of a person and store them in a named folder."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not access the camera.")
        return

    name = input("Enter the person's name: ").strip()
    person_path = os.path.join(data_dir, name)
    os.makedirs(person_path, exist_ok=True)

    print(f"Capturing faces for {name}. Please look at the camera.")

    face_captured = 0
    while face_captured < 5:
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(100, 100))

        for (x, y, w, h) in faces:
            face_img = gray[y:y+h, x:x+w]
            face_file = os.path.join(person_path, f"face_{face_captured + 1}.jpg")
            cv2.imwrite(face_file, face_img)

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"Face {face_captured + 1}", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            face_captured += 1
            break  # Only capture one face per frame

        cv2.imshow("Face Capture", frame)
        cv2.waitKey(500)

    cap.release()
    cv2.destroyAllWindows()
    print(f"Captured {face_captured} faces for {name}.")

def train_recognizer():
    """Train recognizer on all stored faces."""
    images, labels = [], []
    label_names = sorted(os.listdir(data_dir))

    for label, name in enumerate(label_names):
        person_path = os.path.join(data_dir, name)
        if not os.path.isdir(person_path):
            continue

        for filename in os.listdir(person_path):
            if filename.endswith(".jpg"):
                img_path = os.path.join(person_path, filename)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                images.append(img)
                labels.append(label)

    if images:
        recognizer.train(images, np.array(labels))
        recognizer.save(trainer_file)
        print("Training complete.")
    else:
        print("No images found to train.")

def recognize_faces():
    """Recognize faces in real-time from webcam."""
    if not os.path.exists(trainer_file):
        print("Error: Recognizer not trained yet.")
        return

    recognizer.read(trainer_file)
    label_names = sorted(os.listdir(data_dir))

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not access the camera.")
        return

    print("Press 'q' to exit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(100, 100))

        for (x, y, w, h) in faces:
            face_img = gray[y:y+h, x:x+w]
            label, confidence = recognizer.predict(face_img)

            # Debug line
            print(f"Label: {label}, Confidence: {confidence:.2f}")

            if confidence > threshold or label >= len(label_names):
                name = "Unknown"
            else:
                name = label_names[label]

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"{name} ({100 - confidence:.2f}%)", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("Real-Time Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def main_menu():
    while True:
        print("\n1: Capture New Face")
        print("2: Train Recognizer")
        print("3: Start Real-Time Recognition")
        print("4: Exit")

        choice = input("Choose an option: ")
        if choice == "1":
            capture_face()
        elif choice == "2":
            train_recognizer()
        elif choice == "3":
            recognize_faces()
        elif choice == "4":
            print("Exiting.")
            break
        else:
            print("Invalid option.")

# Run the program
main_menu()
