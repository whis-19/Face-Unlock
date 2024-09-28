from Threading import *
def recognize_faces(model_path="trained_knn_model.clf"):
    try:
        with open(model_path, 'rb') as f:
            knn_clf = pickle.load(f)
    except FileNotFoundError:
        print("Model file not found. Please ensure that 'trained_knn_model.clf' exists.")
        return
    except Exception as e:
        print(f"Error loading the model: {e}")
        return

    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("Error: Could not open video stream.")
        return

    print("Starting face recognition. Press 'q' to quit.")

    while True:
        grabbed, image1 = camera.read()
        if not grabbed:
            print("Error: Could not read frame from camera.")
            break

        image = image1[:, :, ::-1]

        face_locations = face_recognition.face_locations(image)

        if face_locations:
            face_encodings = face_recognition.face_encodings(image, face_locations)

            closest_distances = knn_clf.kneighbors(face_encodings, n_neighbors=1)
            are_matches = [closest_distances[0][i][0] <= 0.4 for i in range(len(face_locations))]
            predictions = [
                (pred, loc) if rec else ("unknown", loc)
                for pred, loc, rec in zip(knn_clf.predict(face_encodings), face_locations, are_matches)
            ]

            for name, (top, right, bottom, left) in predictions:
                cv2.rectangle(image1, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(image1, f"{name}", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        else:
            print("No faces found in the current frame.")

        cv2.imshow("Output Image", image1)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting...")
            break

