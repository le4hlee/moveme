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

_last_sign_debug_at = 0.0
SIGN_DEBUG_EVERY_S = 7.0

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
    if (middle_tip[1] - middle_dip[1] < 0.05) and (middle_dip[1] - middle_pip[1] < 0.05) and (middle_pip[1] - middle_mcp[1] < 0.05):
      return True
    else:
      return False
  if finger == "ring":
    if (ring_tip[1] - ring_dip[1] < 0.05) and (ring_dip[1] - ring_pip[1] < 0.05) and (ring_pip[1] - ring_mcp[1] < 0.05):
      return True
    else:
      return False
  if finger == "pinky":
    if (pinky_tip[1] - pinky_dip[1] < 0.05) and (pinky_dip[1] - pinky_pip[1] < 0.05) and (pinky_pip[1] - pinky_mcp[1] < 0.05):
      return True
    else:
      return False
  
  return index_tip[2] > index_dip[2]


def is_hook(finger : str, direction : str, landmarks) -> bool:
  """True for ASL X: index hooked, not a straight G.

  MCP → PIP  up
  PIP → DIP  up-left
  DIP → TIP  down-left
  """
  lm = landmarks.landmark

  index_mcp = _xyz(lm[5])
  index_pip = _xyz(lm[6])
  index_dip = _xyz(lm[7])
  index_tip = _xyz(lm[8])

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
  return False

def fingers_up(fingers: str,landmarks, handedness_label: str) -> list[str]:
  false_count = 0
  names = raised_fingers(landmarks, handedness_label)
  if ("thumb" in fingers and "thumb" not in names) or ("thumb" not in fingers and "thumb" in names):
    false_count += 1
  if ("index" in fingers and "index" not in names) or ("index" not in fingers and "index" in names):
    false_count += 1
  if ("middle" in fingers and "middle" not in names) or ("middle" not in fingers and "middle" in names):
    false_count += 1
  if ("ring" in fingers and "ring" not in names) or ("ring" not in fingers and "ring" in names):
    false_count += 1
  if ("pinky" in fingers and "pinky" not in names) or ("pinky" not in fingers and "pinky" in names):
    false_count += 1
  
  return false_count

def sign_gesture(landmarks, handedness_label: str) -> str:
  """Return the name of the sign gesture based on which fingers are raised"""
  global _last_sign_debug_at
  names = raised_fingers(landmarks, handedness_label)
  lm = landmarks.landmark

  wrist = _xyz(lm[0])

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

  direction = finger_direction(landmarks, handedness_label)
  x_hook = index_is_x_hook(index_mcp, index_pip, index_dip, index_tip)
  
  if fingers_up("thumb", landmarks, handedness_label) == 0 and thumb_tip[1] > thumb_mcp[1] and _dist(thumb_dip, index_mcp) < 0.07:
    return "A"
  if fingers_up("index, middle, ring, pinky", landmarks, handedness_label) == 0 and _dist(thumb_tip, thumb_mcp) > 0 and (_dist(thumb_tip, middle_mcp) < 0.03 or _dist(thumb_tip, index_mcp) < 0.03 or _dist(thumb_tip, ring_mcp) < 0.07):
    return "B"
  if (fingers_up("thumb, index, middle, ring, pinky", landmarks, handedness_label) == 0 or fingers_up("index, middle, ring, pinky", landmarks, handedness_label) == 0):
    if "index right" in direction and "middle right" in direction and "ring right" in direction and "pinky right" in direction and _dist(index_tip, middle_tip) < 0.03 and _dist(middle_tip, ring_tip) < 0.04 and _dist(ring_tip, pinky_tip) < 0.08:
      return "C"
  if (fingers_up("index", landmarks, handedness_label) == 0 or fingers_up("thumb,index", landmarks, handedness_label) == 0) and _dist(thumb_tip, middle_tip) < 0.05 and (_dist(middle_tip, ring_tip) < 0.07 or _dist(ring_tip, pinky_tip) < 0.07) and (index_mcp[1] > index_tip[1]) and not x_hook:
    return "D"
  if fingers_up("thumb, index, middle, ring, pinky", landmarks, handedness_label) == 5  and (_dist(thumb_tip, index_tip) < 0.07 or _dist(thumb_tip, middle_tip) < 0.07 or _dist(thumb_tip, ring_tip) < 0.07 or _dist(thumb_tip, pinky_tip) < 0.07):
    return "E"
  if fingers_up("middle, ring, pinky", landmarks, handedness_label) == 0 and _dist(thumb_tip, index_tip) < 0.07:
    return "F"
  if fingers_up("thumb, index", landmarks, handedness_label) == 0 and "thumb left" in direction and "index left" in direction and _dist(thumb_tip, index_tip) < 0.5 and thumb_tip[1] > index_tip[1] and is_straight("thumb", "left", landmarks) and is_straight("index", "left", landmarks) and not x_hook:
    return "G"
  if fingers_up("index, middle", landmarks, handedness_label) == 0 and "index left" in direction and "middle left" in direction and _dist(index_tip, middle_tip) < 0.1:
    return "H"
  if fingers_up("pinky", landmarks, handedness_label) == 0 and (_dist(thumb_dip, index_mcp) < 0.07 or _dist(thumb_dip, middle_mcp) < 0.07 or _dist(thumb_dip, ring_mcp) < 0.07):
    return "I"
  # J track movement
  if fingers_up("index, middle", landmarks, handedness_label) == 0 and "index up" in direction and "middle up" in direction and _dist(thumb_tip, index_mcp) < 0.07 and _dist(thumb_tip, middle_mcp) < 0.07:
    return "K"
  if fingers_up("thumb, index", landmarks, handedness_label) == 0 and "thumb right" in direction and "index up" in direction:
    return "L"
  # M and N (thumb is invisible)
  if fingers_up("thumb, index, middle, ring, pinky", landmarks, handedness_label) == 5 and "thumb right" in direction and _dist(thumb_tip, index_tip) < 0.07 and _dist(thumb_tip, middle_tip) < 0.07 and _dist(thumb_tip, ring_tip) < 0.07 and _dist(thumb_tip, pinky_tip) < 0.07:
    return "O"
  if fingers_up("index, middle", landmarks, handedness_label) == 0 and "index left" in direction and "middle left" in direction and "middle down" in direction and _dist(thumb_tip, index_pip) < 0.07 and _dist(thumb_pip, middle_mcp) < 0.07:
    return "P"
  if fingers_up("index", landmarks, handedness_label) == 0 and "thumb left" in direction and "thumb down" in direction and"index left" in direction and "index down" in direction and _dist(thumb_tip, index_tip) < 0.3:
    return "Q"
  if fingers_up("index, middle", landmarks, handedness_label) == 0 and _dist(thumb_tip, index_mcp) < 0.07 and _dist(index_dip, middle_dip) < 0.03 and index_tip[0] < middle_tip[0]:
    return "R"
  if fingers_up("thumb, index, middle, ring, pinky", landmarks, handedness_label) == 5 and (_dist(thumb_tip, index_dip) < 0.07 or _dist(thumb_tip, middle_dip) < 0.07) and (index_tip[2] > index_dip[2] and middle_tip[2] > middle_dip[2]):
    return "S" # must find a way to differentiate between E and S
  if fingers_up("index, middle, ring, pinky", landmarks, handedness_label) >= 4 and "thumb up" in direction and (_dist(thumb_dip, index_mcp) < 0.1 or _dist(thumb_dip, index_pip) < 0.1) and (_dist(thumb_dip, middle_mcp) < 0.1 or _dist(thumb_dip, middle_pip) < 0.1):
    return "T"
  if fingers_up("index, middle", landmarks, handedness_label) == 0 and _dist(thumb_tip, ring_mcp) < 0.07 and _dist(index_tip, middle_tip) < 0.07:
    return "U"
  if fingers_up("index, middle", landmarks, handedness_label) == 0 and _dist(thumb_tip, ring_mcp) < 0.07 and _dist(index_tip, middle_tip) < 0.2:
    return "V"
  if fingers_up("index, middle, ring", landmarks, handedness_label) == 0 and _dist(thumb_tip, pinky_mcp) < 0.07 and _dist(index_tip, middle_tip) < 0.2 and _dist(middle_tip, ring_tip) < 0.2:
    return "W"
  if fingers_up("index", landmarks, handedness_label) == 0 and "index left" in direction and _dist(thumb_tip, middle_tip) < 0.07 and is_straight("index", "left", landmarks) == False and is_hook("index", "left", landmarks) == False:
    return "X"
  # else:
    # print("fingers_up(index, middle, ring, pinky, landmarks, handedness_label)", fingers_up("index, middle, ring, pinky", landmarks, handedness_label))
  return "unknown"
  
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
  if "index" in names and (index_tip[0] > index_mcp[0]):
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
    
    print("Camera on. Hold up a finger. Press Q to quit. Press K to print landmark xyz. Press C with an open hand to calibrate.\n")

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
          sign = None

          if result.multi_hand_landmarks and result.multi_handedness:
            hand_landmarks = result.multi_hand_landmarks[0]
            lm = hand_landmarks.landmark
            current_lm = lm
            handedness_label = result.multi_handedness[0].classification[0].label
            names = raised_fingers(hand_landmarks, handedness_label)
            message = format_fingers(names)
            gesture = hand_gesture(hand_landmarks, handedness_label)
            direction = finger_direction(hand_landmarks, handedness_label)
            touching = finger_touching(hand_landmarks, handedness_label)
            sign = sign_gesture(hand_landmarks, handedness_label)

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
            movement_until = 0.0

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
            if sign is not None:
              print(f"sign_gesture: {sign}", flush=True)
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
          if sign is not None:
            cv2.putText(
              frame,
              f"sign_gesture: {sign}",
              (20, 280),
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
