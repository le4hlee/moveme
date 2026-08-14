"""
Day 3 — Finger detector

Reads your laptop camera, detects which fingers are held up
(thumb, index, middle, ring, pinky), and prints the result
to the terminal whenever it changes.

Run:
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    python detect_fingers.py

Press Q in the camera window (or Ctrl+C in the terminal) to quit.
"""

from __future__ import annotations

import math
import sys
import time

import cv2
import mediapipe as mp

# MediaPipe Hands landmark indices:
#   0      wrist
#   1–4    thumb   (CMC, MCP, IP, TIP)
#   5–8    index   (MCP, PIP, DIP, TIP)
#   9–12   middle  (MCP, PIP, DIP, TIP)
#   13–16  ring    (MCP, PIP, DIP, TIP)
#   17–20  pinky   (MCP, PIP, DIP, TIP)
FINGER_TIPS = {
    "thumb": 4,
    "index": 8,
    "middle": 12,
    "ring": 16,
    "pinky": 20,
}

# PIP (or IP for the thumb) — the joint we compare the tip against
FINGER_PIPS = {
    "thumb": 3,
    "index": 6,
    "middle": 10,
    "ring": 14,
    "pinky": 18,
}

FINGER_MCPS = {
    "thumb": 2,
    "index": 5,
    "middle": 9,
    "ring": 13,
    "pinky": 17,
}


def _xyz(landmark) -> tuple[float, float, float]:
    return (landmark.x, landmark.y, landmark.z)


def _sub(
    a: tuple[float, float, float], b: tuple[float, float, float]
) -> tuple[float, float, float]:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _dot(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _dist(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return math.sqrt(_dot(_sub(a, b), _sub(a, b)))


def _norm(v: tuple[float, float, float]) -> tuple[float, float, float]:
    length = math.sqrt(_dot(v, v)) or 1e-9
    return (v[0] / length, v[1] / length, v[2] / length)


def thumb_is_raised(lm, handedness_label: str) -> bool:
    """True only when the thumb is actually held out, not merely resting.

    Comparing the thumb tip to its own joints in image-y is a bad test:
    a resting thumb still points somewhat "up" along the side of the hand,
    so that rule fires all the time. Instead we measure the thumb against
    the palm:

    - Raised next to the other fingers: the tip reaches past the knuckles,
      about as far as an extended index PIP.
    - Stuck out to the side: the tip is clearly farther from the palm
      centerline than the index knuckle is.
    """
    wrist = _xyz(lm[0])
    tip = _xyz(lm[4])
    ip = _xyz(lm[3])
    index_mcp = _xyz(lm[5])
    middle_mcp = _xyz(lm[9])
    pinky_mcp = _xyz(lm[17])

    palm_width = _dist(index_mcp, pinky_mcp) or 1e-6
    # Wrist → middle knuckle is the "up the palm" direction, independent
    # of which way the camera is tilted.
    along_axis = _norm(_sub(middle_mcp, wrist))

    def along(point: tuple[float, float, float]) -> float:
        return _dot(_sub(point, wrist), along_axis)

    def away_from_centerline(point: tuple[float, float, float]) -> float:
        v = _sub(point, wrist)
        nearest = (
            wrist[0] + _dot(v, along_axis) * along_axis[0],
            wrist[1] + _dot(v, along_axis) * along_axis[1],
            wrist[2] + _dot(v, along_axis) * along_axis[2],
        )
        return _dist(point, nearest)

    palm_length = along(middle_mcp) or 1e-6

    # Thumb tucked across the palm (peace sign / four fingers): the tip
    # sits near the pinky side, not out as its own digit.
    if _dist(tip, pinky_mcp) < 0.6 * palm_width:
        return False

    # Pose A: thumb raised like the other fingers. The tip must pass the
    # knuckles by a real margin — resting along the index does not.
    # Use the knuckles (not the index PIP) as the reference so a curled
    # index finger cannot make a resting thumb look "long enough".
    raised_with_fingers = along(tip) > along(middle_mcp) + 0.35 * palm_length

    # Pose B: thumb abducted sideways (open hand / "five").
    stuck_out = away_from_centerline(tip) > away_from_centerline(index_mcp) + 0.28 * palm_width
    if handedness_label == "Right":
        sideways_x = lm[4].x < lm[3].x - 0.03
    else:
        sideways_x = lm[4].x > lm[3].x + 0.03
    raised_to_side = stuck_out and sideways_x and _dist(tip, pinky_mcp) > _dist(ip, pinky_mcp)

    return raised_with_fingers or raised_to_side


def raised_fingers(landmarks, handedness_label: str) -> list[str]:
    """Return the names of fingers that are currently held up.

    For index/middle/ring/pinky: the tip is "up" if it sits higher on
    the image than the PIP and MCP joints (image y grows downward).

    For the thumb: see thumb_is_raised — it handles both a sideways
    stick-out and a thumb raised alongside the other fingers.
    """
    lm = landmarks.landmark
    up: list[str] = []

    if thumb_is_raised(lm, handedness_label):
        up.append("thumb")

    for name in ("index", "middle", "ring", "pinky"):
        tip = lm[FINGER_TIPS[name]]
        pip = lm[FINGER_PIPS[name]]
        mcp = lm[FINGER_MCPS[name]]
        if tip.y < pip.y and tip.y < mcp.y:
            up.append(name)

    return up


def format_fingers(names: list[str]) -> str:
    if not names:
        return "none"
    return ", ".join(names)


def main() -> int:
    hands_solution = mp.solutions.hands
    drawer = mp.solutions.drawing_utils
    styles = mp.solutions.drawing_styles

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open the camera. Check that no other app is using it.", file=sys.stderr)
        return 1

    # Prefer a modest resolution so detection stays snappy.
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    last_message = None
    last_print_at = 0.0
    print("Camera on. Hold up a finger. Press Q to quit.\n")

    with hands_solution.Hands(
        static_image_mode=False,
        max_num_hands=1,
        model_complexity=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.6,
    ) as hands:
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    print("Lost the camera feed.", file=sys.stderr)
                    break

                # Mirror so moving your right hand matches the right side of the window.
                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb.flags.writeable = False
                result = hands.process(rgb)
                rgb.flags.writeable = True

                message = "no hand"
                if result.multi_hand_landmarks and result.multi_handedness:
                    hand_landmarks = result.multi_hand_landmarks[0]
                    handedness_label = result.multi_handedness[0].classification[0].label
                    names = raised_fingers(hand_landmarks, handedness_label)
                    message = format_fingers(names)

                    drawer.draw_landmarks(
                        frame,
                        hand_landmarks,
                        hands_solution.HAND_CONNECTIONS,
                        styles.get_default_hand_landmarks_style(),
                        styles.get_default_hand_connections_style(),
                    )

                # Print only when the detected fingers change, so the
                # terminal is not flooded with identical lines.
                now = time.time()
                if message != last_message and (now - last_print_at) > 0.15:
                    print(message, flush=True)
                    last_message = message
                    last_print_at = now

                overlay = f"finger: {message}   (Q to quit)"
                cv2.putText(
                    frame,
                    overlay,
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 220, 0),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow("Day 3 — finger detector", frame)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    break
        except KeyboardInterrupt:
            print("\nStopped.")
        finally:
            cap.release()
            cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    sys.exit(main())