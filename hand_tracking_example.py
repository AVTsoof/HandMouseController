import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)  # webcam No.0

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

# fps
ptime = 0
ctime = 0

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # for: hands.process()
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks is not None:
        for handlms in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(img, handlms, mpHands.HAND_CONNECTIONS)

            # draw numbers on hand
            for id, lm in enumerate(handlms.landmark):
                # lm contains x,y,z data in ratio of the image
                h, w, c = img.shape
                xpx = round(lm.x * w)
                ypx = round(lm.y * h)
                cv2.putText(img, str(id), (xpx, ypx), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)

    # fps
    ctime = time.time()
    fps = round(1.0 / (ctime - ptime))
    ptime = ctime
    cv2.putText(img, str(fps), (10, 50), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 255), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
