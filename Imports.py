import cv2
import mediapipe as mp
import pyautogui
import threading
import time
import keyboard
import face_recognition


mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)
