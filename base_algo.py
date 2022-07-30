from re import L
import time
import cv2
from matplotlib.pyplot import axis
import numpy as np
import mediapipe as mp
import autopy
import enum
import math

mpHands = mp.solutions.hands
hands = mpHands.Hands(min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

class State(enum.Enum):
    DISABLED = enum.auto()
    START = enum.auto()
    ENABLED = enum.auto()
    END = enum.auto()

class Posture(enum.Enum):
    ACTIVATE = enum.auto()
    IDLE = enum.auto()
    LEFT_CLICK = enum.auto()
    RIGHT_CLICK = enum.auto()
    MIDDLE_CLICK = enum.auto()


class Gesture(enum.Enum):
    IDLE = enum.auto()


def filter_positions(positions, last_positions):
    # TODO: implement filter wiggling fingers
    # ...
    return positions

def get_posture(positions) -> Posture:
    # TODO: implement
    
    dist = get_positions_dist(positions, 4,20)
    print(dist)
    if dist < 0.07:
        return Posture.ACTIVATE

    return None

def get_gesture(positions) -> Gesture:
    # TODO: implement
    return None


def move_mouse(positions, last_positions):
    if positions is None:
        return

def get_positions_dist(positions, idx1, idx2):
    p1 = positions.landmark[idx1]
    p2 = positions.landmark[idx2]

    p1 = np.array((p1.x, p1.y, p1.z))
    p2 = np.array((p2.x, p2.y, p2.z))

    squared_dist = np.sum((p1-p2)**2, axis=0)
    dist = np.sqrt(squared_dist)
    return dist

def draw_hand(img, positions):
    mpDraw.draw_landmarks(img, positions, mpHands.HAND_CONNECTIONS)

    # draw numbers on hand
    for id, lm in enumerate(positions.landmark):
        # lm contains x,y,z data in ratio of the image
        h, w, c = img.shape
        xpx = round(lm.x * w)
        ypx = round(lm.y * h)
        cv2.putText(img, str(id), (xpx, ypx), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)

def draw_fps(img, prev_time):
    curr_time = time.time()
    fps = round(1.0 / (curr_time - prev_time))
    prev_time = curr_time
    cv2.putText(img, str(fps), (10, 50), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 255), 3)
    return prev_time

def draw_state(img, state: State):
    cv2.putText(img, state.name, (100, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)


def main():
    t0 = time.time()
    cap = cv2.VideoCapture(0)  # webcam No.0
    state = State.DISABLED
    last_positions = None
    last_posture = None
    fps_time = 0

    while True:
        # TODO: Use threads / processes
        success, img = cap.read()  # read image
        if not success:
            continue
        
        # get fingers positions
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # for: hands.process()
        hands_positions = hands.process(imgRGB).multi_hand_landmarks
        
        if hands_positions is None:
            continue

        positions = hands_positions[0]  # use the first hand
        positions = filter_positions(positions, last_positions)
        posture = get_posture(positions)
        gesture = get_gesture(positions)

        # state select
        timeout = lambda t : (time.time() - t > 3)
        
        if state == State.DISABLED:
            if posture == Posture.ACTIVATE:
                if timeout(t0):
                    state = State.START
            else:
                t0 = time.time()
        
        elif state == State.START:
            if posture != Posture.ACTIVATE:
                state = State.ENABLED
        
        elif state == State.ENABLED:
            if posture == Posture.ACTIVATE:
                if timeout(t0):    
                    state = State.END
            else:
                t0 = time.time()

        elif state == State.END:
            if posture != Posture.ACTIVATE:
                state = State.DISABLED

        # state run
        if state == State.ENABLED:
            if posture == Posture.IDLE:
                move_mouse(positions, last_positions)
            elif posture == Posture.LEFT_CLICK:
                pass
            elif posture == Posture.RIGHT_CLICK:
                pass
            elif posture == Posture.MIDDLE_CLICK:
                pass
        
        last_positions = positions
        last_posture = posture

        draw_state(img, state)
        draw_hand(img, positions)
        fps_time = draw_fps(img, fps_time)
        cv2.imshow("Image", img)

        # exit if any key is pressed
        if cv2.waitKey(1) != -1:
            break
        
            
if __name__ == '__main__':
    main()
