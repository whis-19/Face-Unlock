from Threading import *


# Register a face
def register_face():
    cam = cv2.VideoCapture(0)
    print("Please look at the camera to register your face...")

    while True:
        ret, frame = cam.read()
        if not ret:
            print("Can't get the frame")
            break
        
        cv2.imshow('Register Face', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('s'):
            cv2.imwrite('authorized_face.jpg', frame)
            print("Face saved!")
            break

    cam.release()
    cv2.destroyAllWindows()

# Function for face lock (authentication)
def face_lock():
    registered_face_image = face_recognition.load_image_file("authorized_face.jpg")
    registered_face_encoding = face_recognition.face_encodings(registered_face_image)[0]

    cam = cv2.VideoCapture(0)
    print("Face lock is starting...")

    while True:
        ret, frame = cam.read()
        if not ret:
            print("Can't get the frame")
            break

        rgb_frame = frame[:, :, ::-1]
        face_locations = face_recognition.face_locations(rgb_frame)
        if face_locations:
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        else:
            print("No faces found!")

       

        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces([registered_face_encoding], face_encoding)

            if True in matches:
                print("Access Granted!")
                cv2.putText(frame, "Access Granted", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                print("Access Denied!")
                cv2.putText(frame, "Access Denied", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        if face_locations:
            for location in face_locations:
                top, right, bottom, left = location
                cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)

        cv2.imshow('Face Lock', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()



