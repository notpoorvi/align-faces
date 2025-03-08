import cv2
import dlib
import numpy as np
from imutils import face_utils

detector = dlib.get_frontal_face_detector()

predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def align_face(image, face):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    landmarks = predictor(gray, face)
    landmarks = face_utils.shape_to_np(landmarks)
    
    left_eye = landmarks[36:42]
    right_eye = landmarks[42:48]
    
    left_eye_center = left_eye.mean(axis=0).astype("int")
    right_eye_center = right_eye.mean(axis=0).astype("int")
    
    dy = right_eye_center[1] - left_eye_center[1]
    dx = right_eye_center[0] - left_eye_center[0]
    
    angle = np.degrees(np.arctan2(dy, dx)) # - 180

    dist = np.sqrt((dx ** 2) + (dy ** 2))
    output_size = (400, 400)
    
    desired_eye_distance = output_size[0] * 0.01
    scale = desired_eye_distance / dist
    
    eyes_center = (
        int((left_eye_center[0] + right_eye_center[0]) // 2),
        int((left_eye_center[1] + right_eye_center[1]) // 2)
    )
    M = cv2.getRotationMatrix2D(eyes_center, angle, scale)
  
    tX = output_size[0] * 0.5
    tY = output_size[1] * 0.4
    M[0, 2] += (tX - eyes_center[0])
    M[1, 2] += (tY - eyes_center[1])
    
    aligned_face = cv2.warpAffine(image, M, output_size)
    return M 

def align_frame(frame, M, output_size=(400, 400)):
    return cv2.warpAffine(frame, M, output_size)

cap = cv2.VideoCapture("IMG_2955.MOV")
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
fps = int(cap.get(cv2.CAP_PROP_FPS))
out = cv2.VideoWriter("aligned_video.mp4", fourcc, fps, (400, 400))

M = None
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    if M is None:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        if len(faces) > 0:
            M = align_face(frame, faces[0])

    if M is not None:
        aligned = align_frame(frame, M)
        out.write(aligned)
        cv2.imshow("Aligned Video", aligned)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

# image = cv2.imread("img.jpg")
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# faces = detector(gray)

# aligned = align_face(image, faces[0])
# cv2.imshow('aligned face', aligned)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
