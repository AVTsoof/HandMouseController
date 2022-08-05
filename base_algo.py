# TODO: mouse moves relative to user's distance from screen

import time
import cv2
from matplotlib.pyplot import axis
import numpy as np
import enum
import math
from scipy.signal import butter, lfilter

import mediapipe as mp
# enum of hand joints; see: https://google.github.io/mediapipe/images/mobile/hand_landmarks.png
from mediapipe.python.solutions.hands import HandLandmark as HL

import autopy  # for mouse control

mpHands = mp.solutions.hands
hands = mpHands.Hands(min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

SCREEN_W, SCREEN_H = autopy.screen.size()
CAM_W, CAM_H = 1280, 720

ACTIVATE_TIMEOUT = 5  # sec
NUM_JOINTS = 21


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


def butter_lowpass_filter(data, cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = lfilter(b, a, data)
    return y


def calc_cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def filter_joints(seq, fps):
    # TODO: not good...
    cutoff = fps / 3
    seq = np.array(seq)  # [frame, joint, xyz]

    for j in range(NUM_JOINTS):
        joint_frames = seq[:, j, :]
        x, y, z = tuple(joint_frames.T)
        x, y, z = \
            butter_lowpass_filter(x, cutoff, fs=fps), \
            butter_lowpass_filter(y, cutoff, fs=fps), \
            butter_lowpass_filter(z, cutoff, fs=fps)
        joint_frames = np.array([x, y, z]).T
        seq[:, j, :] = joint_frames
    return list(seq)


def get_posture(joints) -> Posture:
    # TODO: implement

    # use wrist as reference point
    f0 = joints[HL.WRIST]

    # Posture.ACTIVATE
    f1, f2 = joints[HL.THUMB_TIP], joints[HL.PINKY_TIP]

    sim = calc_cosine_sim(f1 - f0, f2 - f0)
    if sim > 0.9:
        return Posture.ACTIVATE

    return Posture.IDLE


def get_gesture(postures_seq) -> Gesture:
    # TODO: implement
    return None


def get_hand_bounds(joints: np.ndarray):
    # calc over joint axis, results for min\max of xyz for all
    joints_min = np.min(joints, axis=0)
    joints_max = np.max(joints, axis=0)
    bounds = {
        'x0': joints_min[0], 'x1': joints_max[0],
        'y0': joints_min[1], 'y1': joints_max[1],
        'z0': joints_min[2], 'z1': joints_max[2],
    }
    return bounds


def move_mouse(joints):
    if joints is None:
        return

    xyz = joints[HL.INDEX_FINGER_TIP]

    x = np.interp(xyz[0], (0, 1), (0, SCREEN_W))
    y = np.interp(xyz[1], (0, 1), (0, SCREEN_W))

    dx = x
    dy = y

    dx = dx if dx > 0 else 0
    dx = dx if dx <= SCREEN_W else SCREEN_W - 1

    dy = dy if dy > 0 else 0
    dy = dy if dy <= SCREEN_H else SCREEN_H - 1

    autopy.mouse.move(dx, dy)


def draw_hand(img, positions, bounds):
    mpDraw.draw_landmarks(img, positions, mpHands.HAND_CONNECTIONS)
    x0, y0 = round(CAM_W * bounds['x0']), round(CAM_H * bounds['y0'])
    x1, y1 = round(CAM_W * bounds['x1']), round(CAM_H * bounds['y1'])
    cv2.rectangle(img, (x0, y0), (x1, y1), (0, 0, 255), thickness=2)

    # draw numbers on hand
    for id, lm in enumerate(positions.landmark):
        # lm contains x,y,z data in ratio of the image
        h, w, c = img.shape
        xpx = round(lm.x * w)
        ypx = round(lm.y * h)
        cv2.putText(img, str(id), (xpx, ypx), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)


def draw_fps(img, fps):
    cv2.putText(img, str(fps), (10, 50), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 255), 3)


def draw_state(img, state: State):
    cv2.putText(img, state.name, (100, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)


def state_select(state, time_passed, posture):
    timeout = lambda t, T: (time.time() - t > T)

    if state == State.DISABLED:
        if posture == Posture.ACTIVATE:
            if timeout(time_passed, ACTIVATE_TIMEOUT):
                state = State.START
        else:
            time_passed = time.time()

    elif state == State.START:
        time_passed = time.time()
        if posture != Posture.ACTIVATE:
            state = State.ENABLED

    elif state == State.ENABLED:
        if posture == Posture.ACTIVATE:
            if timeout(time_passed, ACTIVATE_TIMEOUT):
                state = State.END
        else:
            time_passed = time.time()

    elif state == State.END:
        time_passed = time.time()
        if posture != Posture.ACTIVATE:
            state = State.DISABLED

    return state, time_passed


def states_run(state, joints_seq, postures_seq):
    joints = joints_seq[-1]
    posture = postures_seq[-1]

    if state == State.ENABLED:
        if posture == Posture.IDLE:
            move_mouse(joints)
        elif posture == Posture.LEFT_CLICK:
            pass
        elif posture == Posture.RIGHT_CLICK:
            pass
        elif posture == Posture.MIDDLE_CLICK:
            pass


def main():
    time_passed = time.time()
    cap = cv2.VideoCapture(0)  # webcam No.0
    cap.set(3, CAM_W)
    cap.set(4, CAM_H)
    state = State.ENABLED  # TODO: state = State.DISABLED
    fps_ptime = time.time()

    max_frames = 30
    joints_seq = []
    postures_seq = []
    fps_seq = []

    while True:
        # TODO: Use threads / processes
        # exit display if any key is pressed
        if cv2.waitKey(1) != -1:
            break

        # fps
        fps_ctime = time.time()
        fps = round(1.0 / (fps_ctime - fps_ptime))
        fps_ptime = fps_ctime

        # fps mean
        if len(fps_seq) == max_frames:
            fps_seq = fps_seq[1:]  # remove old frame
        fps_seq.append(fps)
        mfps = int(np.mean(fps_seq).round())

        success, img = cap.read()  # read image
        if not success:
            continue
        img = cv2.flip(img, flipCode=1)

        # get hand positions
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # for: hands.process()
        hands_positions = hands.process(imgRGB).multi_hand_landmarks

        positions = None

        if hands_positions is not None:
            positions = hands_positions[0]  # use the first hand
            # convert to array
            joints = np.array([(p.x, p.y, p.z) for p in positions.landmark])

            if len(joints_seq) == max_frames:
                joints_seq = joints_seq[1:]  # remove old frame
            joints_seq.append(joints)

            hand_bounds = get_hand_bounds(joints)

            # TODO: joints_seq = filter_joints(joints_seq, mfps)

            posture = get_posture(joints)
            if len(postures_seq) == max_frames:
                postures_seq = postures_seq[1:]  # remove old frame
            postures_seq.append(posture)

            gesture = get_gesture(postures_seq)

            state, time_passed = state_select(state, time_passed, posture)
            states_run(state, joints_seq, postures_seq)

        draw_state(img, state)
        if positions:
            draw_hand(img, positions, hand_bounds)
        draw_fps(img, mfps)
        cv2.imshow("Image", img)


if __name__ == '__main__':
    main()
