import cv2
import os
import numpy as np

# Paths for storing trained faces and recognizer file
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
data_dir = "stored_faces"
trainer_file = "trainer.yml"
threshold = 50  # Confidence threshold for face recognition (below this, face is "unknown")

# Ensure face storage directory exists
os.makedirs(data_dir, exist_ok=True)

# Initialize face recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

def capture_face():
    """Capture multiple face images of a person and ask for their name."""
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not access the camera.")
        return

    name = input("Enter the person's name: ").strip()

    # Create directory for the person if not exists
    person_path = os.path.join(data_dir, name)
    os.makedirs(person_path, exist_ok=True)

    print(f"Position the person in front of the camera for {name}...")

    # Give some time for the camera to adjust and focus
    print("Waiting for camera to adjust...")
    cv2.waitKey(2000)  # Wait for 2 seconds

    face_captured = 0
    while face_captured < 5:
        ret, frame = cap.read()

        if not ret:
            print("Failed to capture frame. Retrying...")
            continue  # Retry capturing the frame if failed

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

        if len(faces) > 0:
            # Capture the first face detected
            x, y, w, h = faces[0]
            face_img = gray[y:y+h, x:x+w]
            face_filename = os.path.join(person_path, f"face_{face_captured + 1}.jpg")
            cv2.imwrite(face_filename, face_img)

            # Draw a rectangle around the detected face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"Capturing Face {face_captured + 1}...", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.imshow("Face Capture", frame)
            cv2.waitKey(2000)  # Pause for 2 seconds to display confirmation

            face_captured += 1
            print(f"Face {face_captured} captured.")
        else:
            print("No faces detected, trying again...")

    cap.release()
    cv2.destroyAllWindows()
    print(f"Captured {face_captured} faces for {name}.")

def train_recognizer():
    """ Train the face recognizer with images stored in the data directory. """
    images, labels = [], []
    
    # Loop through each folder in the data directory and load the images
    for label, name in enumerate(os.listdir(data_dir)):
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
        print("Training completed.")
    else:
        print("No faces found to train.")

def recognize_faces():
    """ Start real-time face recognition. """
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not access the camera.")
        return

    recognizer.read(trainer_file)
    print("Real-time recognition started. Press 'q' to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame.")
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))
        
        for (x, y, w, h) in faces:
            face_img = gray[y:y+h, x:x+w]
            label, confidence = recognizer.predict(face_img)

            # Check if the confidence is below a threshold (indicating "unknown")
            if confidence < threshold:
                name = "Unknown"
                confidence_text = f"Confidence: {100 - confidence:.2f}%"
            else:
                name = os.listdir(data_dir)[label]
                confidence_text = f"Confidence: {100 - confidence:.2f}%"
            
            # Draw a rectangle and name on the face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"{name} {confidence_text}", 
                        (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("Real-Time Face Recognition", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):  # Press 'q' to exit the recognition loop
            break

    cap.release()
    cv2.destroyAllWindows()

def main_menu():
    """ Main menu to control face capture, training, and recognition. """
    while True:
        print("\n1: Capture a New Face")
        print("2: Train Recognizer")
        print("3: Start Real-Time Recognition")
        print("4: Exit Program")

        choice = input("Choose an option: ")

        if choice == "1":
            capture_face()
        elif choice == "2":
            train_recognizer()
        elif choice == "3":
            recognize_faces()
        elif choice == "4":
            print("Exiting program.")
            break
        else:
            print("Invalid choice, try again.")

# Start the program
main_menu()
