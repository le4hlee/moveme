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


def finger_is_extended(lm, name: str) -> bool:
    """True when a finger is stretched out, no matter which way the hand faces.

    Image-y (tip above knuckle) only works if fingers point at the top of
    the camera. When the hand points down, an extended tip is *below* the
    knuckle. Distance from the wrist still grows as the finger uncurls.
    """
    wrist = _xyz(lm[0])
    tip = _xyz(lm[FINGER_TIPS[name]])
    pip = _xyz(lm[FINGER_PIPS[name]])
    return _dist(wrist, tip) > _dist(wrist, pip)


def _screen_direction(tip, mcp, min_len: float) -> str | None:
    """Label which way a finger points on the camera: up/down/left/right."""
    dx, dy, _dz = _sub(tip, mcp)
    if math.hypot(dx, dy) < min_len:
        return None
    if abs(dx) > abs(dy):
        return "left" if dx < 0 else "right"
    # Image y grows downward, so smaller y is toward the top of the window.
    return "up" if dy < 0 else "down"


def raised_fingers(landmarks, handedness_label: str) -> list[str]:
    """Return the names of fingers that are currently extended.

    Extension is measured from the wrist, not the top of the screen, so a
    hand pointing down still counts open fingers as open.
    """
    lm = landmarks.landmark
    up: list[str] = []

    if thumb_is_raised(lm, handedness_label):
        up.append("thumb")

    for name in ("index", "middle", "ring", "pinky"):
        if finger_is_extended(lm, name):
            up.append(name)

    return up

def hand_gesture(landmarks, handedness_label: str) -> str:
    """Return the name of the hand gesture based on which fingers are raised"""
    names = raised_fingers(landmarks, handedness_label)
    if not names:
        return "fist"
    if "thumb"  in names and "index"  in names and "middle" in names and "ring"  in names and "pinky" in names:
        return "hi"
    if "thumb"  in names and "index" in names and "middle" not in names and "ring"  not in names and "pinky" not in names:
        return "um actually"
    if "thumb" not in names and "index" in names and "middle" in names and "ring" not in names and "pinky" not in names:
        return "peace"
    if "thumb" not in names and "index" in names and "middle" not in names and "ring" not in names and "pinky" in names:
        return "rock and roll"
    if "thumb" in names and "index" in names and "middle" not in names and "ring" not in names and "pinky" in names:
        return "rock and roll..?"
    if "thumb" in names and "index" not in names and "middle" not in names and "ring" not in names and "pinky" in names:
        return "shaka"
    if "thumb" in names and "index" not in names and "middle" not in names and "ring" not in names and "pinky" not in names:
        return "thumbs up"
    if "index" in names and "thumb" not in names and "middle" not in names and "ring" not in names and "pinky" not in names:
        return "point up"
    if "middle" in names and "thumb" not in names and "index" not in names and "ring" not in names and "pinky" not in names:
        return "hey that's rude"
    if "ring" in names and "thumb" not in names and "index" not in names and "middle" not in names and "pinky" not in names:
        return "where's my ring?"
    if "pinky" in names and "thumb" not in names and "index" not in names and "middle" not in names and "ring" not in names:
        return "pinky up"
    # Any other mix of raised fingers has no named pose
    # — still return a string so main() can concatenate it.
    return "unknown"

def finger_direction(landmarks, handedness_label: str) -> str:
    """Return which way each extended finger points on the camera."""
    lm = landmarks.landmark
    names = raised_fingers(landmarks, handedness_label)
    palm_width = _dist(_xyz(lm[5]), _xyz(lm[17])) or 1e-6
    min_len = 0.2 * palm_width

    total_direction = ""

    thumb_tip = _xyz(lm[4])
    index_tip = _xyz(lm[8])
    middle_tip = _xyz(lm[12])
    ring_tip = _xyz(lm[16])
    pinky_tip = _xyz(lm[20])

    thumb_mcp = _xyz(lm[1])
    index_mcp = _xyz(lm[5])
    middle_mcp = _xyz(lm[9])
    ring_mcp = _xyz(lm[13])
    pinky_mcp = _xyz(lm[17])

    
    # _xyz() returns (x, y, z), so use [1] for y — not .y.
    # Image y grows downward: smaller y = higher on screen = pointing up.
    if "thumb" in names and (thumb_tip[1] < thumb_mcp[1]):
        total_direction += "thumbs up, "

    if "thumb" in names and (thumb_tip[1] > thumb_mcp[1]):
        total_direction += "thumbs down, "
    
    if "index" in names and (index_tip[1] < index_mcp[1]):
        total_direction += "index up, "
    
    if "index" in names and (index_tip[1] > index_mcp[1]):
        total_direction += "index down, "
    
    if "middle" in names and (middle_tip[1] < middle_mcp[1]):
        total_direction += "middle up, "
    
    if "middle" in names and (middle_tip[1] > middle_mcp[1]):
        total_direction += "middle down, "
    
    if "ring" in names and (ring_tip[1] < ring_mcp[1]):
        total_direction += "ring up, "
    
    if "ring" in names and (ring_tip[1] > ring_mcp[1]):
        total_direction += "ring down, "
    
    if "pinky" in names and (pinky_tip[1] < pinky_mcp[1]):
        total_direction += "pinky up, "
    
    if "pinky" in names and (pinky_tip[1] > pinky_mcp[1]):
        total_direction += "pinky down, "

    return total_direction
    
def finger_touching(landmarks, handedness_label: str) -> str:
    """Return the fingers that are touching each other"""
    lm = landmarks.landmark
    names = raised_fingers(landmarks, handedness_label)

    thumb_tip = _xyz(lm[4])
    index_tip = _xyz(lm[8])
    middle_tip = _xyz(lm[12])
    ring_tip = _xyz(lm[16])
    pinky_tip = _xyz(lm[20])

    total_touching = ""
    if "thumb" in names and "index" in names and _dist(thumb_tip, index_tip) < 0.07:
        total_touching += "thumb and index, "
    if "thumb" in names and "middle" in names and _dist(thumb_tip, middle_tip) < 0.07:
        total_touching += "thumb and middle, "
    if "thumb" in names and "ring" in names and _dist(thumb_tip, ring_tip) < 0.07:
        total_touching += "thumb and ring, "
    if "thumb" in names and "pinky" in names and _dist(thumb_tip, pinky_tip) < 0.07:
        total_touching += "thumb and pinky, "
    
    if "index" in names and "middle" in names and _dist(index_tip, middle_tip) < 0.07:
        total_touching += "index and middle, "
    if "index" in names and "ring" in names and _dist(index_tip, ring_tip) < 0.07:
        total_touching += "index and ring, "
    if "index" in names and "pinky" in names and _dist(index_tip, pinky_tip) < 0.07:
        total_touching += "index and pinky, "

    if "middle" in names and "ring" in names and _dist(middle_tip, ring_tip) < 0.07:
        total_touching += "middle and ring, "
    if "middle" in names and "pinky" in names and _dist(middle_tip, pinky_tip) < 0.07:
        total_touching += "middle and pinky, "
    
    if "ring" in names and "pinky" in names and _dist(ring_tip, pinky_tip) < 0.07:
        total_touching += "ring and pinky, "

    return total_touching or "none"

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
    prev_wrist = None
    prev_thumb_tip = None
    prev_index_tip = None
    prev_middle_tip = None
    prev_ring_tip = None
    prev_pinky_tip = None

    prev_thumb_mcp = None
    prev_index_mcp = None
    prev_middle_mcp = None
    prev_ring_mcp = None
    prev_pinky_mcp = None
    
    movement_until = 0.0
    movement_hold_s = 0.7
    
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
                gesture = None
                direction = None
                touching = None
                movement_x = None
                movement_y = None

                if result.multi_hand_landmarks and result.multi_handedness:
                    hand_landmarks = result.multi_hand_landmarks[0]
                    lm = hand_landmarks.landmark
                    handedness_label = result.multi_handedness[0].classification[0].label
                    names = raised_fingers(hand_landmarks, handedness_label)
                    message = format_fingers(names)
                    gesture = hand_gesture(hand_landmarks, handedness_label)
                    direction = finger_direction(hand_landmarks, handedness_label)
                    touching = finger_touching(hand_landmarks, handedness_label)

                    drawer.draw_landmarks(
                        frame,
                        hand_landmarks,
                        hands_solution.HAND_CONNECTIONS,
                        styles.get_default_hand_landmarks_style(),
                        styles.get_default_hand_connections_style(),
                    )

                    wrist = _xyz(lm[0])
                    thumb_tip = _xyz(lm[4])
                    index_tip = _xyz(lm[8])
                    middle_tip = _xyz(lm[12])
                    ring_tip = _xyz(lm[16])
                    pinky_tip = _xyz(lm[20])
                    thumb_mcp = _xyz(lm[1])
                    index_mcp = _xyz(lm[5])
                    middle_mcp = _xyz(lm[9])
                    ring_mcp = _xyz(lm[13])
                    pinky_mcp = _xyz(lm[17])

                    # if prev_wrist is not None and _dist(wrist, prev_wrist) > 0.05:
                    #     print("wrist moved", flush=True)
                    # if prev_thumb_tip is not None and _dist(thumb_tip, prev_thumb_tip) > 0.05:
                    #     print("thumb tip moved", flush=True)
                    # if prev_index_tip is not None and _dist(index_tip, prev_index_tip) > 0.05:
                    #     print("index tip moved", flush=True)
                    
                    now = time.time()
                    if (
                        prev_index_tip is not None
                        and _dist(index_tip, prev_index_tip) > 0.05
                        and "index" in names
                    ):
                        movement_until = now + movement_hold_s
                    if now < movement_until and "index" in names:
                        gesture = "nahhh"
                    
                    if prev_index_tip is not None and _dist(index_tip, prev_index_tip) > 0.05 and prev_index_tip[0] > index_tip[0]:
                        print("index tip moved left", flush=True)
                        now = time.time()
                        movement_until = now + movement_hold_s
                        if now < movement_until and "index" in names:
                            movement_x = "index tip left"
                    if prev_index_tip is not None and _dist(index_tip, prev_index_tip) > 0.05 and prev_index_tip[0] < index_tip[0]:
                        print("index tip moved right", flush=True)
                        now = time.time()
                        movement_until = now + movement_hold_s
                        if now < movement_until and "index" in names:
                            movement_x = "index tip right"
                    
                    if prev_index_tip is not None and _dist(index_tip, prev_index_tip) > 0.05 and prev_index_tip[1] > index_tip[1]:
                        print("index tip moved up", flush=True)
                        now = time.time()
                        movement_until = now + movement_hold_s
                        if now < movement_until and "index" in names:
                            movement_y = "index tip up"
                    if prev_index_tip is not None and _dist(index_tip, prev_index_tip) > 0.05 and prev_index_tip[1] < index_tip[1]:
                        print("index tip moved down", flush=True)
                        now = time.time()
                        movement_until = now + movement_hold_s
                        if now < movement_until and "index" in names:
                            movement_y = "index tip down"

                    # Always save this frame so the next frame has a previous point.
                    prev_wrist = wrist
                    prev_thumb_tip = thumb_tip
                    prev_index_tip = index_tip
                    prev_middle_tip = middle_tip
                    prev_ring_tip = ring_tip
                    prev_pinky_tip = pinky_tip
                    prev_thumb_mcp = thumb_mcp
                    prev_index_mcp = index_mcp
                    prev_middle_mcp = middle_mcp
                    prev_ring_mcp = ring_mcp
                    prev_pinky_mcp = pinky_mcp
                else:
                    prev_wrist = None
                    prev_thumb_tip = None
                    prev_index_tip = None
                    prev_middle_tip = None
                    prev_ring_tip = None
                    prev_pinky_tip = None
                    prev_thumb_mcp = None
                    prev_index_mcp = None
                    prev_middle_mcp = None
                    prev_ring_mcp = None
                    prev_pinky_mcp = None
                    nahh_until = 0.0

                # Print only when the detected fingers or gesture change, so the
                # terminal is not flooded with identical lines.
                now = time.time()
                status = (message, gesture, direction, touching, movement_x, movement_y)
                if status != last_message and (now - last_print_at) > 0.15:
                    print(message, flush=True)
                    if gesture is not None:
                        print(f"gesture: {gesture}", flush=True)
                    if direction is not None:
                        print(f"direction: {direction}", flush=True)
                    if touching is not None:
                        print(f"touching: {touching}", flush=True)
                    if movement_x is not None:
                        print(f"movement_x: {movement_x}", flush=True)
                    if movement_y is not None:
                        print(f"movement_y: {movement_y}", flush=True)
                    last_message = status
                    last_print_at = now

                cv2.putText(
                    frame,
                    f"finger: {message}   (Q to quit)",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 220, 0),
                    2,
                    cv2.LINE_AA,
                )
                if gesture is not None:
                    cv2.putText(
                        frame,
                        f"gesture: {gesture}",
                        (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 220, 0),
                        2,
                        cv2.LINE_AA,
                    )
                if direction is not None:
                    cv2.putText(
                        frame,
                        f"direction: {direction}",
                        (20, 120),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 220, 0),
                        2,
                        cv2.LINE_AA,
                    )
                if touching is not None:
                    cv2.putText(
                        frame,
                        f"touching: {touching}",
                        (20, 160),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 220, 0),
                        2,
                        cv2.LINE_AA,
                    )
                if movement_x is not None:
                    cv2.putText(
                        frame,
                        f"movement_x: {movement_x}",
                        (20, 200),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 220, 0),
                        2,
                        cv2.LINE_AA,
                    )
                if movement_y is not None:
                    cv2.putText(
                        frame,
                        f"movement_y: {movement_y}",
                        (20, 240),
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
