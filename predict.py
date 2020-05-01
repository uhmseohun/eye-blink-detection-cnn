from tensorflow.keras.models import load_model
from imutils.face_utils import shape_to_np, rect_to_bb
from dataset import IMAGE_SIZE
import cv2, dlib
import time
import numpy as np

model = load_model('trained_model/model.h5')
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('dlib_predictor/shape_predictor_68_face_landmarks.dat')

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while (True):
  ret, frame = cap.read()
  frame = cv2.resize(frame, dsize=(0, 0), fx=0.7, fy=0.7, interpolation=cv2.INTER_AREA)
  frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

  faces = detector(frame, 0)

  if (len(faces) == 0):
    print('Face not found')
    continue

  face = faces[0]

  (x, y, w, h) = rect_to_bb(face)

  landmarks = predictor(frame, face)
  landmarks = shape_to_np(landmarks)

  (lx, ly) = (landmarks[42][0], landmarks[43][1])
  (lw, lh) = (landmarks[45][0] - lx, landmarks[47][1] - ly)
  (lx, ly, lw, lh) = (lx-10, ly-10, lw+30, lh+20)

  (rx, ry) = (landmarks[36][0], landmarks[37][1])
  (rw, rh) = (landmarks[39][0] - rx, landmarks[41][1] - ry)
  (rx, ry, rw, rh) = (rx-10, ry-10, rw+30, rh+20)

  eyes = [
    frame[ly:ly+lh, lx:lx+lw],
    frame[ry:ry+rh, rx:rx+rw]
  ]

  data = []
  for eye in eyes:
    eye = cv2.resize(eye, dsize=IMAGE_SIZE[::-1])
    eye = np.array(eye).reshape(*IMAGE_SIZE, 1).astype(np.float32)
    eye = eye / 255.
    data.append(eye)
  data = np.array(data)

  result = model.predict(data)

  print({
    'left': result[0][0] > 0.5,
    'right': result[1][0] > 0.5
  })

  if cv2.waitKey(33) > 0:
    break
