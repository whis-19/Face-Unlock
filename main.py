import cv2
import face_recognition
import numpy as np

# Function to capture and save the authorized user's face during setup
def register_face():
    cam = cv2.VideoCapture(0)
    print("Look into the camera for face registration...")

    while True:
        ret, frame = cam.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        # Display the frame while registering face
        cv2.imshow('Register Face', frame)
        
        # Press 's' to save and register the face
        if cv2.waitKey(1) & 0xFF == ord('s'):
            # Save the frame as the authorized user's face
            cv2.imwrite('authorized_face.jpg', frame)
            print("Face registered successfully!")
            break

    cam.release()
    cv2.destroyAllWindows()

# Function to perform face lock (authentication)
def face_lock():
    # Load the authorized face
    registered_face_image = face_recognition.load_image_file("authorized_face.jpg")
    registered_face_encoding = face_recognition.face_encodings(registered_face_image)[0]

    # Initialize camera for live face detection
    cam = cv2.VideoCapture(0)
    print("Starting Face Lock...")

    while True:
        ret, frame = cam.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Convert the frame to RGB as face_recognition uses RGB images
        rgb_frame = frame[:, :, ::-1]

        # Find all face locations and encodings in the current frame
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for face_encoding in face_encodings:
            # Compare the face with the registered face
            matches = face_recognition.compare_faces([registered_face_encoding], face_encoding)

            if True in matches:
                print("Access Granted!")
                cv2.putText(frame, "Access Granted", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                print("Access Denied!")
                cv2.putText(frame, "Access Denied", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Display the camera frame with access result
        cv2.imshow('Face Lock', frame)

        # Press 'q' to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()

# Main function
if __name__ == "__main__":
    print("1. Register Face")
    print("2. Start Face Lock")
    choice = input("Enter your choice (1/2): ")

    if choice == '1':
        register_face()
    elif choice == '2':
        face_lock()
    else:
        print("Invalid choice")
