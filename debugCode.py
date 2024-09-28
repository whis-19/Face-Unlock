import cv2
import face_recognition

# Initialize video capture
video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    rgb_frame = frame[:, :, ::-1]  # Convert BGR to RGB

    # Detect faces
    face_locations = face_recognition.face_locations(rgb_frame)
    print("Detected face locations:", face_locations)

    # Process encodings
    if face_locations:
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        print("Face encodings:", face_encodings)

    # Show the video feed
    cv2.imshow('Video', frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
video_capture.release()
cv2.destroyAllWindows()
