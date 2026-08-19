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
Press C with an open hand to calibrate finger lengths.
"""

from __future__ import annotations

import math
import sys
import time
from unittest import skip

import cv2
import mediapipe as mp

# for mouse and keyboard actions
_pyautogui_error = None
try:
  import pyautogui
  pyautogui.FAILSAFE = False
  pyautogui.PAUSE = 0
except Exception as exc:
  pyautogui = None
  _pyautogui_error = exc

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


# Filled by calibrate() when you press C. Not written to disk.
# CALIBRATED_LM[i] = (x, y, z) for landmark i (0–20)
# CALIBRATED_SEGMENTS[j] = (mcp_pip, pip_dip, dip_tip)
#   j order: thumb, index, middle, ring, pinky
#   thumb has no DIP, so its pip_dip is 0.0
CALIBRATED_LM: list[tuple[float, float, float]] | None = None
CALIBRATED_SEGMENTS: list[tuple[float, float, float]] | None = None
FINGER_CAL_ORDER = ("thumb", "index", "middle", "ring", "pinky")


def calibrate(lm) -> list[tuple[float, float, float]]:
  """Save this frame's landmarks and per-bone lengths as in-memory arrays.

  Hold an open hand (all fingers stretched) and press C. Later frames
  compare each finger to these lengths, scaled by palm width so moving
  closer to or farther from the camera does not throw the scale off.
  """
  global CALIBRATED_LM, CALIBRATED_SEGMENTS
  landmarks = [_xyz(lm[i]) for i in range(21)]
  segments: list[tuple[float, float, float]] = []
  for name in FINGER_CAL_ORDER:
    tip_i = FINGER_TIPS[name]
    if name == "thumb":
      mcp_pip = _dist(landmarks[2], landmarks[3])
      pip_dip = 0.0
      dip_tip = _dist(landmarks[3], landmarks[4])
    else:
      mcp = landmarks[tip_i - 3]
      pip = landmarks[tip_i - 2]
      dip = landmarks[tip_i - 1]
      tip = landmarks[tip_i]
      mcp_pip = _dist(mcp, pip)
      pip_dip = _dist(pip, dip)
      dip_tip = _dist(dip, tip)
    segments.append((mcp_pip, pip_dip, dip_tip))

  CALIBRATED_LM = landmarks
  CALIBRATED_SEGMENTS = segments
  return segments


def finger_is_extended(lm, name: str) -> bool:
  """True when a finger is stretched out, no matter which way the hand faces.

  Image-y (tip above knuckle) only works if fingers point at the top of
  the camera. When the hand points down, an extended tip is *below* the
  knuckle. Distance from the wrist still grows as the finger uncurls.
  """
  wrist = _xyz(lm[0])
  tip = _xyz(lm[FINGER_TIPS[name]])
  pip = _xyz(lm[FINGER_PIPS[name]])
  if CALIBRATED_LM is None:
    return _dist(wrist, tip) > _dist(wrist, pip)
  palm_width = _dist(_xyz(lm[5]), _xyz(lm[17])) or 1e-6
  current = _dist(wrist, tip) / palm_width
  cal_palm = _dist(CALIBRATED_LM[5], CALIBRATED_LM[17]) or 1e-6
  full = _dist(CALIBRATED_LM[0], CALIBRATED_LM[FINGER_TIPS[name]]) / cal_palm
  return current > 0.85 * full


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
  if (fingers_up("thumb, index, middle, ring, pinky", landmarks, handedness_label) == 0):
    return "hi"
  if (fingers_up("thumb, index", landmarks, handedness_label) == 0 and fingers_down("middle, ring, pinky", landmarks, handedness_label) == 0):
    return "zoom"
  if (fingers_up("thumb, index, middle", landmarks, handedness_label) == 0 and fingers_down("ring, pinky", landmarks, handedness_label) == 0):
    return "slide"
  if (fingers_up("index, middle", landmarks, handedness_label) == 0 and fingers_down("thumb, ring, pinky", landmarks, handedness_label) == 0):
    return "scroll"
  if (fingers_up("index", landmarks, handedness_label) == 0 and fingers_down("thumb, middle, ring, pinky", landmarks, handedness_label) == 0):
    return "click"
  
  # Any other mix of raised fingers has no named pose
  # — still return a string so main() can concatenate it.
  return "unknown"


def is_straight(finger : str, direction : str, landmarks) -> bool:
  lm = landmarks.landmark

  thumb_tip = _xyz(lm[4])
  index_tip = _xyz(lm[8])
  middle_tip = _xyz(lm[12])
  ring_tip = _xyz(lm[16])
  pinky_tip = _xyz(lm[20])

  thumb_dip = _xyz(lm[3])
  index_dip = _xyz(lm[7])
  middle_dip = _xyz(lm[11])
  ring_dip = _xyz(lm[15])
  pinky_dip = _xyz(lm[19])

  thumb_pip = _xyz(lm[2])
  index_pip = _xyz(lm[6])
  middle_pip = _xyz(lm[10])
  ring_pip = _xyz(lm[14])
  pinky_pip = _xyz(lm[18])

  thumb_mcp = _xyz(lm[1])
  index_mcp = _xyz(lm[5])
  middle_mcp = _xyz(lm[9])
  ring_mcp = _xyz(lm[13])
  pinky_mcp = _xyz(lm[17])
  
  if finger == "thumb":
    if direction == "up":
      if (thumb_tip[1] - thumb_dip[1] < 0.05) and (thumb_dip[1] - thumb_pip[1] < 0.05) and (thumb_pip[1] - thumb_mcp[1] < 0.05):
        print("Thumb is straight" + str(thumb_tip[1] - thumb_dip[1]) + " " + str(thumb_dip[1] - thumb_pip[1]) + " " + str(thumb_pip[1] - thumb_mcp[1]))
        return True
      else:
        return False
    if direction == "down":
      if (thumb_tip[1] - thumb_dip[1] < -0.05) and (thumb_dip[1] - thumb_pip[1] < -0.05) and (thumb_pip[1] - thumb_mcp[1] < -0.05):
        print("Thumb is straight" + str(thumb_tip[1] - thumb_dip[1]) + " " + str(thumb_dip[1] - thumb_pip[1]) + " " + str(thumb_pip[1] - thumb_mcp[1]))
        return True
      else:
        return False
    if direction == "left":
      if (thumb_tip[0] - thumb_dip[0] < -0.05) and (thumb_dip[0] - thumb_pip[0] < -0.05) and (thumb_pip[0] - thumb_mcp[0] < -0.05):
        print("Thumb is straight" + str(thumb_tip[1] - thumb_dip[1]) + " " + str(thumb_dip[1] - thumb_pip[1]) + " " + str(thumb_pip[1] - thumb_mcp[1]))
        return True
      else:
        return False
    if direction == "right":
      if (thumb_tip[0] - thumb_dip[0] < 0.05) and (thumb_dip[0] - thumb_pip[0] < 0.05) and (thumb_pip[0] - thumb_mcp[0] < 0.05):
        print("Thumb is straight" + str(thumb_tip[1] - thumb_dip[1]) + " " + str(thumb_dip[1] - thumb_pip[1]) + " " + str(thumb_pip[1] - thumb_mcp[1]))
        return True
      else:
        return False
  if finger == "index":
    if direction == "up":
      if (index_tip[1] - index_dip[1] < 0.05) and (index_dip[1] - index_pip[1] < 0.05) and (index_pip[1] - index_mcp[1] < 0.05):
        print("Index is straight" + str(index_tip[1] - index_dip[1]) + " " + str(index_dip[1] - index_pip[1]) + " " + str(index_pip[1] - index_mcp[1]))
        return True
      else:
        return False
    if direction == "down":
      if (index_tip[1] - index_dip[1] < -0.05) and (index_dip[1] - index_pip[1] < -0.05) and (index_pip[1] - index_mcp[1] < -0.05):
        print("Index is straight" + str(index_tip[1] - index_dip[1]) + " " + str(index_dip[1] - index_pip[1]) + " " + str(index_pip[1] - index_mcp[1]))
        return True
      else:
        return False
    if direction == "left":
      if (index_tip[0] - index_dip[0] < -0.05) and (index_dip[0] - index_pip[0] < -0.05) and (index_pip[0] - index_mcp[0] < -0.05):
        print("Index is straight" + str(index_tip[1] - index_dip[1]) + " " + str(index_dip[1] - index_pip[1]) + " " + str(index_pip[1] - index_mcp[1]))
        return True
      else:
        return False
    if direction == "right":
      if (index_tip[0] - index_dip[0] < 0.05) and (index_dip[0] - index_pip[0] < 0.05) and (index_pip[0] - index_mcp[0] < 0.05):
        print("Index is straight" + str(index_tip[1] - index_dip[1]) + " " + str(index_dip[1] - index_pip[1]) + " " + str(index_pip[1] - index_mcp[1]))
        return True
      else:
        return False
  if finger == "middle":
    if direction == "up":
      if (middle_tip[1] - middle_dip[1] < 0.05) and (middle_dip[1] - middle_pip[1] < 0.05) and (middle_pip[1] - middle_mcp[1] < 0.05):
        print("Middle is straight" + str(middle_tip[1] - middle_dip[1]) + " " + str(middle_dip[1] - middle_pip[1]) + " " + str(middle_pip[1] - middle_mcp[1]))
        return True
      else:
        return False
    if direction == "down":
      if (middle_tip[1] - middle_dip[1] < -0.05) and (middle_dip[1] - middle_pip[1] < -0.05) and (middle_pip[1] - middle_mcp[1] < -0.05):
        print("Middle is straight" + str(middle_tip[1] - middle_dip[1]) + " " + str(middle_dip[1] - middle_pip[1]) + " " + str(middle_pip[1] - middle_mcp[1]))
        return True
      else:
        return False
    if direction == "left":
      if (middle_tip[0] - middle_dip[0] < -0.05) and (middle_dip[0] - middle_pip[0] < -0.05) and (middle_pip[0] - middle_mcp[0] < -0.05):
        print("Middle is straight" + str(middle_tip[1] - middle_dip[1]) + " " + str(middle_dip[1] - middle_pip[1]) + " " + str(middle_pip[1] - middle_mcp[1]))
        return True
      else:
        return False
    if direction == "right":
      if (middle_tip[0] - middle_dip[0] < 0.05) and (middle_dip[0] - middle_pip[0] < 0.05) and (middle_pip[0] - middle_mcp[0] < 0.05):
        print("Middle is straight" + str(middle_tip[1] - middle_dip[1]) + " " + str(middle_dip[1] - middle_pip[1]) + " " + str(middle_pip[1] - middle_mcp[1]))
        return True
      else:
        return False

  if finger == "ring":
    if direction == "up":
      if (ring_tip[1] - ring_dip[1] < 0.05) and (ring_dip[1] - ring_pip[1] < 0.05) and (ring_pip[1] - ring_mcp[1] < 0.05):
        print("Ring is straight" + str(ring_tip[1] - ring_dip[1]) + " " + str(ring_dip[1] - ring_pip[1]) + " " + str(ring_pip[1] - ring_mcp[1]))
        return True
      else:
        return False
    if direction == "down":
      if (ring_tip[1] - ring_dip[1] < -0.05) and (ring_dip[1] - ring_pip[1] < -0.05) and (ring_pip[1] - ring_mcp[1] < -0.05):
        print("Ring is straight" + str(ring_tip[1] - ring_dip[1]) + " " + str(ring_dip[1] - ring_pip[1]) + " " + str(ring_pip[1] - ring_mcp[1]))
        return True
      else:
        return False
    if direction == "left":
      if (ring_tip[0] - ring_dip[0] < -0.05) and (ring_dip[0] - ring_pip[0] < -0.05) and (ring_pip[0] - ring_mcp[0] < -0.05):
        print("Ring is straight" + str(ring_tip[1] - ring_dip[1]) + " " + str(ring_dip[1] - ring_pip[1]) + " " + str(ring_pip[1] - ring_mcp[1]))
        return True
      else:
        return False
    if direction == "right":
      if (index_tip[0] - index_dip[0] < 0.05) and (index_dip[0] - index_pip[0] < 0.05) and (index_pip[0] - index_mcp[0] < 0.05):
        print("Index is straight" + str(index_tip[1] - index_dip[1]) + " " + str(index_dip[1] - index_pip[1]) + " " + str(index_pip[1] - index_mcp[1]))
        return True
      else:
        return False
  if finger == "pinky":
    if direction == "up":
      if (pinky_tip[1] - pinky_dip[1] < 0.05) and (pinky_dip[1] - pinky_pip[1] < 0.05) and (pinky_pip[1] - pinky_mcp[1] < 0.05):
        print("Pinky is straight" + str(pinky_tip[1] - pinky_dip[1]) + " " + str(pinky_dip[1] - pinky_pip[1]) + " " + str(pinky_pip[1] - pinky_mcp[1]))
        return True
      else:
        return False
    if direction == "down":
      if (pinky_tip[1] - pinky_dip[1] < -0.05) and (pinky_dip[1] - pinky_pip[1] < -0.05) and (pinky_pip[1] - pinky_mcp[1] < -0.05):
        print("Pinky is straight" + str(pinky_tip[1] - pinky_dip[1]) + " " + str(pinky_dip[1] - pinky_pip[1]) + " " + str(pinky_pip[1] - pinky_mcp[1]))
        return True
      else:
        return False
    if direction == "left":
      if (pinky_tip[0] - pinky_dip[0] < -0.05) and (pinky_dip[0] - pinky_pip[0] < -0.05) and (pinky_pip[0] - pinky_mcp[0] < -0.05):
        print("Pinky is straight" + str(pinky_tip[1] - pinky_dip[1]) + " " + str(pinky_dip[1] - pinky_pip[1]) + " " + str(pinky_pip[1] - pinky_mcp[1]))
        return True
      else:
        return False
    if direction == "right":
      if (pinky_tip[0] - pinky_dip[0] < 0.05) and (pinky_dip[0] - pinky_pip[0] < 0.05) and (pinky_pip[0] - pinky_mcp[0] < 0.05):
        print("Pinky is straight" + str(pinky_tip[1] - pinky_dip[1]) + " " + str(pinky_dip[1] - pinky_pip[1]) + " " + str(pinky_pip[1] - pinky_mcp[1]))
        return True
      else:
        return False
  
  return index_tip[2] > index_dip[2]


def is_hook(finger : str, direction : str, landmarks) -> bool:
  """Return True if the finger is hooked in the given direction

  MCP → PIP  up
  PIP → DIP  up-left
  DIP → TIP  down-left
  """
  lm = landmarks.landmark

  thumb_mcp = _xyz(lm[1])
  thumb_pip = _xyz(lm[2])
  thumb_dip = _xyz(lm[3])
  thumb_tip = _xyz(lm[4])

  index_mcp = _xyz(lm[5])
  index_pip = _xyz(lm[6])
  index_dip = _xyz(lm[7])
  index_tip = _xyz(lm[8])

  middle_mcp = _xyz(lm[9])
  middle_pip = _xyz(lm[10])
  middle_dip = _xyz(lm[11])
  middle_tip = _xyz(lm[12])

  ring_mcp = _xyz(lm[13])
  ring_pip = _xyz(lm[14])
  ring_dip = _xyz(lm[15])
  ring_tip = _xyz(lm[16])

  pinky_mcp = _xyz(lm[17])
  pinky_pip = _xyz(lm[18])
  pinky_dip = _xyz(lm[19])
  pinky_tip = _xyz(lm[20])

  if finger == "thumb":
    if direction == "left":
      mcp_to_pip = thumb_pip[1] < thumb_mcp[1]
      pip_to_dip = thumb_dip[0] < thumb_pip[0] and thumb_dip[1] < thumb_pip[1]
      dip_to_tip = thumb_tip[0] < thumb_dip[0] and thumb_tip[1] > thumb_dip[1]
      if mcp_to_pip and pip_to_dip and dip_to_tip:
        return True
      else:
        return False  
    if direction == "right":
      mcp_to_pip = index_pip[1] > index_mcp[1]
      pip_to_dip = index_dip[0] > index_pip[0] and index_dip[1] > index_pip[1]
      dip_to_tip = index_tip[0] > index_dip[0] and index_tip[1] < index_dip[1]
      if mcp_to_pip and pip_to_dip and dip_to_tip:
        return True
      else:
        return False
  if finger == "index":
    if direction == "left":
      mcp_to_pip = index_pip[1] < index_mcp[1]
      pip_to_dip = index_dip[0] < index_pip[0] and index_dip[1] < index_pip[1]
      dip_to_tip = index_tip[0] < index_dip[0] and index_tip[1] > index_dip[1]
      if mcp_to_pip and pip_to_dip and dip_to_tip:
        return True
      else:
        return False  
    if direction == "right":
      mcp_to_pip = index_pip[1] > index_mcp[1]
      pip_to_dip = index_dip[0] > index_pip[0] and index_dip[1] > index_pip[1]
      dip_to_tip = index_tip[0] > index_dip[0] and index_tip[1] < index_dip[1]
      if mcp_to_pip and pip_to_dip and dip_to_tip:
        return True
      else:
        return False
  if finger == "middle":
    if direction == "left":
      mcp_to_pip = middle_pip[1] < middle_mcp[1]
      pip_to_dip = middle_dip[0] < middle_pip[0] and middle_dip[1] < middle_pip[1]
      dip_to_tip = middle_tip[0] < middle_dip[0] and middle_tip[1] > middle_dip[1]
      if mcp_to_pip and pip_to_dip and dip_to_tip:
        return True
      else:
        return False  
    if direction == "right":
      mcp_to_pip = middle_pip[1] > middle_mcp[1]
      pip_to_dip = middle_dip[0] > middle_pip[0] and middle_dip[1] > middle_pip[1]
      dip_to_tip = middle_tip[0] > middle_dip[0] and middle_tip[1] < middle_dip[1]
      if mcp_to_pip and pip_to_dip and dip_to_tip:
        return True
      else:
        return False
  if finger == "ring":
    if direction == "left":
      mcp_to_pip = ring_pip[1] < ring_mcp[1]
      pip_to_dip = ring_dip[0] < ring_pip[0] and ring_dip[1] < ring_pip[1]
      dip_to_tip = ring_tip[0] < ring_dip[0] and ring_tip[1] > ring_dip[1]
      if mcp_to_pip and pip_to_dip and dip_to_tip:
        return True
      else:
        return False  
    if direction == "right":
      mcp_to_pip = ring_pip[1] > ring_mcp[1]
      pip_to_dip = ring_dip[0] > ring_pip[0] and ring_dip[1] > ring_pip[1]
      dip_to_tip = ring_tip[0] > ring_dip[0] and ring_tip[1] < ring_dip[1]
      if mcp_to_pip and pip_to_dip and dip_to_tip:
        return True
      else:
        return False
  if finger == "pinky":
    if direction == "left":
      mcp_to_pip = pinky_pip[1] < pinky_mcp[1]
      pip_to_dip = pinky_dip[0] < pinky_pip[0] and pinky_dip[1] < pinky_pip[1]
      dip_to_tip = pinky_tip[0] < pinky_dip[0] and pinky_tip[1] > pinky_dip[1]
      if mcp_to_pip and pip_to_dip and dip_to_tip:
        return True
      else:
        return False  
    if direction == "right":
      mcp_to_pip = pinky_pip[1] > pinky_mcp[1]
      pip_to_dip = pinky_dip[0] > pinky_pip[0] and pinky_dip[1] > pinky_pip[1]
      dip_to_tip = pinky_tip[0] > pinky_dip[0] and pinky_tip[1] < pinky_dip[1]
      if mcp_to_pip and pip_to_dip and dip_to_tip:
        return True
      else:
        return False
  return False

def fingers_up(fingers: str, landmarks, handedness_label: str = "Right") -> int:
  """Return the count of fingers that are not up within the given list of fingers
    This reduces the lenght of if conditons and also could be used to check if a a list of
    fingers are up.
  """
  false_count = 0
  names = raised_fingers(landmarks, handedness_label)
  if ("thumb" in fingers and "thumb" not in names):
    false_count += 1
  if ("index" in fingers and "index" not in names):
    false_count += 1
  if ("middle" in fingers and "middle" not in names):
    false_count += 1
  if ("ring" in fingers and "ring" not in names):
    false_count += 1
  if ("pinky" in fingers and "pinky" not in names):
    false_count += 1
  
  return false_count

def fingers_down(fingers: str, landmarks, handedness_label: str = "Right") -> int:
  """Return the count of fingers that are not down within the given list of fingers"""
  false_count = 0
  names = raised_fingers(landmarks, handedness_label)
  if ("thumb" in fingers and "thumb" in names):
    false_count += 1
  if ("index" in fingers and "index" in names):
    false_count += 1
  if ("middle" in fingers and "middle" in names):
    false_count += 1
  if ("ring" in fingers and "ring" in names):
    false_count += 1
  if ("pinky" in fingers and "pinky" in names):
    false_count += 1
  return false_count

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
  if (thumb_tip[1] < thumb_mcp[1]):
    total_direction += "thumb up, "
  if (thumb_tip[1] > thumb_mcp[1]):
    total_direction += "thumb down, "
  if (thumb_tip[0] > thumb_mcp[0]):
    total_direction += "thumb right, "
  if (thumb_tip[0] < thumb_mcp[0]):
    total_direction += "thumb left, "

  if (index_tip[1] < index_mcp[1]):
    total_direction += "index up, "
  if (index_tip[1] > index_mcp[1]):
    total_direction += "index down, "
  if (index_tip[0] > index_mcp[0]):
    total_direction += "index right, "
  if (index_tip[0] < index_mcp[0]):
    total_direction += "index left, "

  if (middle_tip[1] < middle_mcp[1]):
    total_direction += "middle up, "
  if (middle_tip[1] > middle_mcp[1]):
    total_direction += "middle down, "
  if (middle_tip[0] > middle_mcp[0]):
    total_direction += "middle right, "
  if (middle_tip[0] < middle_mcp[0]):
    total_direction += "middle left, "

  if (ring_tip[1] < ring_mcp[1]):
    total_direction += "ring up, "
  if (ring_tip[1] > ring_mcp[1]):
    total_direction += "ring down, "
  if (ring_tip[0] > ring_mcp[0]):
    total_direction += "ring right, "
  if (ring_tip[0] < ring_mcp[0]):
    total_direction += "ring left, "

  if (pinky_tip[1] < pinky_mcp[1]):
    total_direction += "pinky up, "
  if (pinky_tip[1] > pinky_mcp[1]):
    total_direction += "pinky down, "
  if (pinky_tip[0] > pinky_mcp[0]):
    total_direction += "pinky right, "
  if (pinky_tip[0] < pinky_mcp[0]):
    total_direction += "pinky left, "

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

def detect_hand(landmarks, handedness_label: str) -> str:
  """Return the hand gesture based on the landmarks"""
  lm = landmarks.landmark

  if lm[1].x < lm[17].x:
    return "left"
  if lm[1].x > lm[17].x:
    return "right"
  return "unknown"

def format_fingers(names: list[str]) -> str:
    if not names:
      return "none"
    return ", ".join(names)


def perform_action(name: str) -> None:
  """Print the action and send it to the focused window."""
  print(f"action: {name}", flush=True)
  if pyautogui is None:
    return
  try:
    if name == "zoom in":
      pyautogui.hotkey("command", "=")
    elif name == "zoom out":
      pyautogui.hotkey("command", "-")
    elif name == "slide left":
      pyautogui.press("left")
    elif name == "slide right":
      pyautogui.press("right")
    elif name == "slide up":
      pyautogui.press("up")
    elif name == "slide down":
      pyautogui.press("down")
    elif name == "scroll up":
      pyautogui.scroll(40)
    elif name == "scroll down":
      pyautogui.scroll(-40)
    elif name == "click":
      pyautogui.click()
    elif name == "double click":
      pyautogui.doubleClick()
  except Exception as exc:
    print(f"could not perform {name}: {exc}", flush=True)


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

    prev_gesture = ""
    prev_gesture_time = 0.0
    last_fist_at = -999.0
    last_hi_at = -999.0
    fist_open_count = 0
    open_fist_count = 0
    prev_fist_to_hi = 0.0
    prev_hi_to_fist = 0.0
    recording = False

    now_click = -10.0
    last_click_at = -10.0

    last_action_name = ""
    last_action_at = 0.0
    prev_touching = None
    prev_pinch = None

    print("Camera on. Fist then open hand to start controlling. Open hand then fist to stop.")
    print("Press Q : quit. Press K : print landmark xyz. Press C with an open hand to calibrate.")
    print(f"python: {sys.executable}", flush=True)
    if pyautogui is None:
      print(
        f"pyautogui unavailable ({_pyautogui_error}). "
        "Gestures will be detected, but clicks/scrolls will not run. "
        "Fix: pip install pyautogui\n",
        flush=True,
      )
    else:
      print(
        "Mouse/keyboard control ready. Click the app you want to control "
        "(Safari, Notes, …) so it is in front of the camera window.\n",
        flush=True,
      )

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
          current_lm = None

          if result.multi_hand_landmarks and result.multi_handedness:
            hand_landmarks = result.multi_hand_landmarks[0]
            lm = hand_landmarks.landmark
            current_lm = lm
            handedness_label = result.multi_handedness[0].classification[0].label
            names = raised_fingers(hand_landmarks, handedness_label)
            message = ""
            gesture = hand_gesture(hand_landmarks, handedness_label)
            direction = finger_direction(hand_landmarks, handedness_label)
            touching = finger_touching(hand_landmarks, handedness_label)

            if (prev_gesture == "" and gesture != "unknown"):
              prev_gesture = gesture
              prev_gesture_time = time.time()
            elif (prev_gesture != gesture) and (gesture != "unknown"):
              print(f"prev_gesture: {prev_gesture}, gesture: {gesture}", flush=True)
              print("time : ", time.time(), "prev_gesture_time : ", prev_gesture_time, flush=True)

              # Start recording/tracking the hand gestures
              if recording == False and (time.time() - prev_gesture_time > 0.5) and prev_gesture == "fist" and gesture == "hi":
                prev_fist_to_hi = time.time()
                if fist_open_count == 0:
                  open_fist_count += 1
                  print("open_fist = 1\n", flush=True)
                if (time.time() - prev_fist_to_hi < 1):
                  print("Start recording now:\n", flush=True)
                  recording = True
                  open_fist_count = 0
                  last_action_name = "start recording"

              # Stop recording/tracking the hand gestures
              if recording == True and (time.time() - prev_gesture_time > 0.5) and prev_gesture == "hi" and gesture == "fist":
                prev_hi_to_fist = time.time()
                if open_fist_count == 0:
                  open_fist_count += 1
                if (time.time() - prev_hi_to_fist < 1):
                  print("Stop recording now:\n", flush=True)
                  recording = False
                  fist_open_count = 0
                  last_action_name = "stop recording"

              # Click is a pose change (point with index). Enter it again
              # within 0.6s after leaving it for a double click.
              if recording and gesture == "click":
                now_click = time.time()
                if now_click - last_click_at < 0.6:
                  perform_action("double click")
                  last_action_name = "double click"
                else:
                  perform_action("click")
                  last_action_name = "click"
                last_click_at = now_click

              prev_gesture = gesture
              prev_gesture_time = time.time()

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
            
            # if now < movement_until and fingers_up("index", hand_landmarks) == 0:
            #   gesture = "nahhh"
            
            if prev_index_tip is not None and _dist(index_tip, prev_index_tip) > 0.05 and prev_index_tip[0] > index_tip[0]:
              # print("index tip moved left", flush=True)
              now = time.time()
              movement_until = now + movement_hold_s
              if now < movement_until and "index" in names:
                movement_x = "index tip left"
            
            if prev_index_tip is not None and _dist(index_tip, prev_index_tip) > 0.05 and prev_index_tip[0] < index_tip[0]:
              # print("index tip moved right", flush=True)
              now = time.time()
              movement_until = now + movement_hold_s
              if now < movement_until and "index" in names:
                movement_x = "index tip right"
            
            if prev_index_tip is not None and _dist(index_tip, prev_index_tip) > 0.05 and prev_index_tip[1] > index_tip[1]:
              # print("index tip moved up", flush=True)
              now = time.time()
              movement_until = now + movement_hold_s
              if now < movement_until and "index" in names:
                movement_y = "index tip up"
            if prev_index_tip is not None and _dist(index_tip, prev_index_tip) > 0.05 and prev_index_tip[1] < index_tip[1]:
              # print("index tip moved down", flush=True)
              now = time.time()
              movement_until = now + movement_hold_s
              if now < movement_until and "index" in names:
                movement_y = "index tip down"

            # Zoom / slide / scroll need this frame's pinch and motion, so they
            # run while you HOLD the pose — not only on the frame it appears.
            palm_width = _dist(index_mcp, pinky_mcp) or 1e-6
            pinch = _dist(thumb_tip, index_tip) / palm_width
            if recording:
              now_act = time.time()
              mode = gesture if gesture not in (None, "unknown") else prev_gesture
              if mode == "zoom" and now_act - last_action_at > 0.35:
                was_pinched = (
                  prev_touching is not None
                  and "thumb and index" in prev_touching
                )
                is_pinched = touching is not None and "thumb and index" in touching
                if is_pinched and not was_pinched:
                  perform_action("zoom in")
                  last_action_name = "zoom in"
                  last_action_at = now_act
                elif was_pinched and not is_pinched:
                  perform_action("zoom out")
                  last_action_name = "zoom out"
                  last_action_at = now_act
                elif prev_pinch is not None:
                  if prev_pinch > 0.45 and pinch <= 0.35:
                    perform_action("zoom in")
                    last_action_name = "zoom in"
                    last_action_at = now_act
                  elif prev_pinch < 0.55 and pinch >= 0.75:
                    perform_action("zoom out")
                    last_action_name = "zoom out"
                    last_action_at = now_act
              if mode == "slide" and now_act - last_action_at > 0.4:
                if movement_x == "index tip left":
                  perform_action("slide left")
                  last_action_name = "slide left"
                  last_action_at = now_act
                elif movement_x == "index tip right":
                  perform_action("slide right")
                  last_action_name = "slide right"
                  last_action_at = now_act
                elif movement_y == "index tip up":
                  perform_action("slide up")
                  last_action_name = "slide up"
                  last_action_at = now_act
                elif movement_y == "index tip down":
                  perform_action("slide down")
                  last_action_name = "slide down"
                  last_action_at = now_act
              if mode == "scroll" and now_act - last_action_at > 0.15:
                if movement_y == "index tip up":
                  perform_action("scroll up")
                  last_action_name = "scroll up"
                  last_action_at = now_act
                elif movement_y == "index tip down":
                  perform_action("scroll down")
                  last_action_name = "scroll down"
                  last_action_at = now_act

            prev_touching = touching
            prev_pinch = pinch

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
            movement_until = 0.0
            prev_touching = None
            prev_pinch = None

          # Print only when the detected fingers or gesture change, so the
          # terminal is not flooded with identical lines.
          now = time.time()
          status = (message, gesture, direction, touching, movement_x, movement_y)
          if status != last_message and (now - last_print_at) > 0.15:
            print(message, flush=True)
            # if gesture is not None:
            #   print(f"gesture: {gesture}", flush=True)
            # if direction is not None:
            #   print(f"direction: {direction}", flush=True)
            # if touching is not None:
            #   print(f"touching: {touching}", flush=True)
            # if movement_x is not None:
            #   print(f"movement_x: {movement_x}", flush=True)
            # if movement_y is not None:
            #   print(f"movement_y: {movement_y}", flush=True)
            last_message = status
            last_print_at = now

          cal_hint = "calibrated" if CALIBRATED_LM is not None else "C to calibrate"
          cv2.putText(
            frame,
            f"finger: {message}   (Q to quit, {cal_hint})",
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
          if last_action_name is not None:
            cv2.putText(
              frame,
              f"last_action_name: {last_action_name}",
              (20, 200),
              cv2.FONT_HERSHEY_SIMPLEX,
              1.0,
              (0, 220, 0),
              2,
              cv2.LINE_AA,
            )
          if recording:
            cv2.putText(
              frame,
              f"recording: True",
              (20, 280),
              cv2.FONT_HERSHEY_SIMPLEX,
              1.0,
              (0, 220, 0),
              2,
              cv2.LINE_AA,
            )
          else:
            cv2.putText(
              frame,
              f"recording: False",
              (20, 280),
              cv2.FONT_HERSHEY_SIMPLEX,
              1.0,
              (0, 220, 0),
              2,
              cv2.LINE_AA,
            )
          cv2.imshow("Hand detector", frame)

          key = cv2.waitKey(1) & 0xFF
          if key in (ord("q"), ord("Q"), 27):
            break
          if key in (ord("k"), ord("K")):
            if current_lm is None:
              print("no hand — no landmarks to print", flush=True)
            else:
              print("\n--- landmarks (x, y, z) ---", flush=True)
              for i, point in enumerate(current_lm):
                print(
                  f"lm[{i}]  x={point.x:.4f}  y={point.y:.4f}  z={point.z:.4f}",
                  flush=True,
                )
              print("--- end ---\n", flush=True)
          if key in (ord("c"), ord("C")):
            if current_lm is None:
              print("no hand — nothing to calibrate", flush=True)
            else:
              cal = calibrate(current_lm)
              print("\n--- calibrated finger lengths (open hand) ---", flush=True)
              for i, point in enumerate(CALIBRATED_LM):
                print(
                  f"lm[{i}]  x={point[0]:.4f}  y={point[1]:.4f}  z={point[2]:.4f}",
                  flush=True,
                )
              print("CALIBRATED_SEGMENTS  [mcp-pip, pip-dip, dip-tip]", flush=True)
              for name, segs in zip(FINGER_CAL_ORDER, cal):
                mcp_pip, pip_dip, dip_tip = segs
                pip_dip_s = f"{pip_dip:.4f}" if name != "thumb" else "n/a"
                print(
                  f"  {name:7s}  [{mcp_pip:.4f}, {pip_dip_s}, {dip_tip:.4f}]",
                  flush=True,
                )
              print("--- end ---\n", flush=True)
      except KeyboardInterrupt:
        print("\nStopped.")
      finally:
        cap.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
  sys.exit(main())
