from Threading import *

def recognize_faces(model_path="trained_knn_model.clf"):
    # Load the KNN model from the file
    try:
        with open(model_path, 'rb') as f:
            knn_clf = pickle.load(f)
    except FileNotFoundError:
        print("Model file not found. Please ensure that 'trained_knn_model.clf' exists.")
        return
    except Exception as e:
        print(f"Error loading the model: {e}")
        return

    # Start video capture
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("Error: Could not open video stream.")
        return

    print("Starting face recognition. Press 'q' to quit.")

    while True:
        # Read the frame from the video stream
        grabbed, image1 = camera.read()
        if not grabbed:
            print("Error: Could not read frame from camera.")
            break

        # Convert the image from BGR (OpenCV format) to RGB (face_recognition format)
        image = image1[:, :, ::-1]

        # Detect faces in the frame
        face_locations = face_recognition.face_locations(image)
        face_landmarks = face_recognition.face_landmarks(image)

        if face_landmarks:
            # Encode the faces found in the image
            face_encodings = face_recognition.face_encodings(image, face_locations)

            # Predict faces using the KNN model
            closest_distances = knn_clf.kneighbors(face_encodings, n_neighbors=1)
            are_matches = [closest_distances[0][i][0] <= 0.4 for i in range(len(face_locations))]

            # Generate predictions
            predictions = [
                (pred, loc) if rec else ("unknown", loc)
                for pred, loc, rec in zip(knn_clf.predict(face_encodings), face_locations, are_matches)
            ]

            # Draw rectangles around faces and label them
            for name, (top, right, bottom, left) in predictions:
                cv2.rectangle(image1, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(image1, f"{name}", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        else:
            print("No landmarks detected")

        # Display the resulting frame with detected faces and names
        cv2.imshow("Output Image", image1)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting...")
            break

    # Release the camera and close all OpenCV windows
    camera.release()
    cv2.destroyAllWindows()

