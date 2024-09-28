from Threading import *
def train_model(train_dir, model_save_path=None, n_neighbors=2, knn_algo='ball_tree', verbose=False):
    encodings = []
    names = []

    person_dirs = os.listdir(train_dir)

    for person in person_dirs:
        person_path = os.path.join(train_dir, person)
        
        if os.path.isdir(person_path):
            pix = os.listdir(person_path)
            for person_img in pix:
                img_path = os.path.join(person_path, person_img)
                face = face_recognition.load_image_file(img_path)
                face_encodings = face_recognition.face_encodings(face)

                if len(face_encodings) > 0:
                    face_enc = face_encodings[0]
                    encodings.append(face_enc)
                    names.append(person)

    knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=knn_algo, weights='distance')
    knn_clf.fit(encodings, names)

    if model_save_path is not None:
        with open(model_save_path, 'wb') as f:
            pickle.dump(knn_clf, f)

    return knn_clf
