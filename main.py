from faceCapture import *
from trainModel import *
from faceUnlock import *

# Main
if __name__ == "__main__":
    print("Instructions")
    ch = input("Enter 1 to capture faces, 2 to recognize faces: ")
    
    if ch == "1":
        print("Enter the person's name:")
        name = input()
        
     
        capture_faces(name)
        print("Training KNN classifier...")
        train_model("TestCases", model_save_path="trained_knn_model.clf", n_neighbors=2)
        
    elif ch == "2":

        recognize_faces()
        
    else:
        print("Wrong choice.\nTerminating")
        exit()
