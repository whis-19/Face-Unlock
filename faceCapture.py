from Threading import*


def capture_faces(name):
    number = 0
    frame_count = 0
    detector = dlib.get_frontal_face_detector()
    folder_name = f"TestCases/{name}"

    if not os.path.exists(folder_name):
        os.mkdir(folder_name)

    camera = cv2.VideoCapture(0)  
    while True:
        if frame_count % 5 == 0:
            print("Keyframe")
            (grabbed, image) = camera.read()
            if not grabbed:
                break

            image = imutils.resize(image, width=500)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            rects = detector(gray, 1)

            for (i, rect) in enumerate(rects):
                (x, y, w, h) = face_utils.rect_to_bb(rect)
                cro = image[y: y + h, x: x + w]
                out_image = cv2.resize(cro, (108, 108))
                frame_path = os.path.join(folder_name, f"{number}.jpg")
                number += 1
                cv2.imwrite(frame_path, out_image)
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            frame_count += 1
        else:
            frame_count += 1
            print("Redundant frame")

        if number > 51:
            break

        cv2.imshow("Output Image", image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.release()
    cv2.destroyAllWindows()


