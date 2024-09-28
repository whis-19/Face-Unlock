from Func import *

# Main
if __name__ == "__main__":
    print("1. Register Face")
    print("2. Start Face Lock")
    choice = input("Choose an option (1/2): ")

    if choice == '1':
        runInThread(register_face)
    elif choice == '2':
        runInThread(face_lock)
    else:
        print("Invalid choice")
