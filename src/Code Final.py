#!/usr/bin/env python3
"""
gesture_controller.py
Webcam gestures + Drone does ORB-SLAM3 mapping
- Gestures: from PC webcam (not the drone stream)
- ORB-SLAM3 reads the drone stream externally; we tail KeyFrameTrajectory.txt
- After takeoff: slow 360° (12x30°) with pauses to help mapping
- Forward/back moves: now NOT blocked by mapping (toggle with SLAM_GATING_ENABLED)
- Altitude control:
    ☝ index_only  -> up <ALTITUDE_STEP_CM>
    ✊ fist        -> down <ALTITUDE_STEP_CM>
"""

import argparse
import cv2
import mediapipe as mp
import time
import socket
import threading
import numpy as np
from collections import deque, defaultdict
from math import hypot
from scipy.spatial.transform import Rotation as R
import os


# ---------- CONFIG ----------
ORB_SLAM_LOG_FILE = r"C:/ORB_SLAM3_Windows/log/KeyFrameTrajectory.txt"
LOCAL_UDP_PORT = 9000
TELLO_IP = "192.168.10.1"
TELLO_CMD_PORT = 8889

# Webcam index for gestures
GESTURE_CAMERA_INDEX = 0   # change if your webcam is not at 0

# Gesture smoothing
DEBOUNCE_HISTORY = 8
STEADY_REQUIRED = 4

# 360° scan settings
SECTOR_SIZE_DEG = 30                 # 12 sectors of 30°
INCREMENTAL_ANGLE = 30               # degrees per rotation step
INCREMENTAL_PAUSE = 1.5              # sec between steps
ROTATION_RETRY_ATTEMPTS = 2
MAPPING_MIN_SECTORS = 10             # require >=10/12 sectors covered
MAPPING_MIN_SAMPLES_PER_SECTOR = 2   # samples per sector

# Command behavior
COMMAND_COOLDOWN = 2.5
MIN_BATTERY_LEVEL = 20
MOVEMENT_DISTANCE_CM = 20            # safer indoors (forward/back)
ALTITUDE_STEP_CM = 20                # up/down step
TELLO_SPEED_CM_S = 10                # slow for SLAM mapping

# Hold spiderman to manual scan
CHECK360_HOLD_REQUIRED = 12

# NEW: disable SLAM gating for forward/back (as requested)
SLAM_GATING_ENABLED = False
# ----------------------------

mp_hands = mp.solutions.hands

# ----------------- Helpers -----------------
def _dist(a, b):
    return hypot(a.x - b.x, a.y - b.y)

def _hand_size(landmarks):
    return _dist(landmarks[mp_hands.HandLandmark.WRIST],
                 landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]) + 1e-6

def _is_extended(lm, tip_idx, pip_idx, hand_size, min_ratio=0.22):
    return (lm[tip_idx].y < lm[pip_idx].y) and (_dist(lm[tip_idx], lm[pip_idx]) > hand_size * min_ratio)

def _fist_folded(lm, hand_size, min_ratio=0.16):
    idx = not _is_extended(lm, mp_hands.HandLandmark.INDEX_FINGER_TIP,  mp_hands.HandLandmark.INDEX_FINGER_PIP,  hand_size, min_ratio)
    mid = not _is_extended(lm, mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_PIP, hand_size, min_ratio)
    rng = not _is_extended(lm, mp_hands.HandLandmark.RING_FINGER_TIP,   mp_hands.HandLandmark.RING_FINGER_PIP,   hand_size, min_ratio)
    pnk = not _is_extended(lm, mp_hands.HandLandmark.PINKY_TIP,         mp_hands.HandLandmark.PINKY_PIP,         hand_size, min_ratio)
    return idx and mid and rng and pnk

def _thumb_extended_and_direction(lm, hand_size):
    tip = lm[mp_hands.HandLandmark.THUMB_TIP]
    mcp = lm[mp_hands.HandLandmark.THUMB_MCP]
    cmc = lm[mp_hands.HandLandmark.THUMB_CMC]
    extended = _dist(tip, cmc) > hand_size * 0.28
    if not extended: return False, None
    if tip.y < mcp.y - hand_size * 0.10: return True, "up"
    if tip.y > mcp.y + hand_size * 0.10: return True, "down"
    return True, None

def _gap(a, b):
    return hypot(a.x - b.x, a.y - b.y)

# ----------------- Gesture classifier -----------------
def get_gesture(hand_landmarks):
    """Returns one of the defined gestures or None (rules tuned for exclusivity)."""
    lm = hand_landmarks.landmark
    hs = _hand_size(lm)

    index_up  = _is_extended(lm, mp_hands.HandLandmark.INDEX_FINGER_TIP,  mp_hands.HandLandmark.INDEX_FINGER_PIP,  hs)
    middle_up = _is_extended(lm, mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_PIP, hs)
    ring_up   = _is_extended(lm, mp_hands.HandLandmark.RING_FINGER_TIP,   mp_hands.HandLandmark.RING_FINGER_PIP,   hs)
    pinky_up  = _is_extended(lm, mp_hands.HandLandmark.PINKY_TIP,         mp_hands.HandLandmark.PINKY_PIP,         hs)
    thumb_ext, thumb_dir = _thumb_extended_and_direction(lm, hs)

    idx_tip = lm[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    mid_tip = lm[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    th_tip  = lm[mp_hands.HandLandmark.THUMB_TIP]

    # PALM (all up) -> takeoff
    if index_up and middle_up and ring_up and pinky_up and thumb_ext: return "palm"
    # STOP (four up, thumb tucked) -> hover/no-op
    if index_up and middle_up and ring_up and pinky_up and not thumb_ext: return "stop"
    # THUMBS UP/DOWN (thumb only + fist)
    if thumb_ext and _fist_folded(lm, hs):
        if thumb_dir == "up" and th_tip.y < lm[mp_hands.HandLandmark.THUMB_IP].y: return "thumbs_up"
        if thumb_dir == "down" and th_tip.y > lm[mp_hands.HandLandmark.THUMB_IP].y: return "thumbs_down"
    # PEACE: index+middle up, others down, V-gap -> cw
    if index_up and middle_up and not ring_up and not pinky_up:
        if _gap(idx_tip, mid_tip) > hs * 0.12: return "peace"
    # OKAY: thumb-index close, others mostly up -> ccw
    if _gap(th_tip, idx_tip) < hs * 0.09 and middle_up and ring_up: return "okay"
    # SPIDERMAN: thumb + index + pinky up; middle & ring down -> manual 360
    if thumb_ext and index_up and pinky_up and not middle_up and not ring_up: return "spiderman"
    # CALL_ME: thumb + pinky up; others down -> land
    if thumb_ext and pinky_up and not index_up and not middle_up and not ring_up: return "call_me"
    # NEW: INDEX_ONLY (index up, others down), thumb can be NEUTRAL or tucked (not strongly up/down)
    if index_up and not middle_up and not ring_up and not pinky_up and (not thumb_ext or thumb_dir is None):
        return "index_only"
    # NEW: FIST (all down incl. thumb) -> descend
    if not index_up and not middle_up and not ring_up and not pinky_up and not thumb_ext: return "fist"
    return None


# --------------- Tello UDP controller ---------------
class TelloController:
    def __init__(self, local_port=LOCAL_UDP_PORT, tello_ip=TELLO_IP, tello_port=TELLO_CMD_PORT):
        self.tello_ip = tello_ip
        self.tello_port = tello_port
        self.tello_address = (self.tello_ip, self.tello_port)
        self.local_ip = ''
        self.local_port = local_port

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((self.local_ip, self.local_port))
        self.response = None
        self.stop_event = threading.Event()
        self.response_thread = threading.Thread(target=self._receive_response, daemon=True)
        self.response_thread.start()

        self.in_flight = False
        self.last_error = None
        self.motor_fault = False
        self.imu_fault = False
        self.last_motor_fault_time = 0.0
        print("✅ TelloController initialized.")

    def _receive_response(self):
        while not self.stop_event.is_set():
            try:
                self.sock.settimeout(1.0)
                data, _ = self.sock.recvfrom(1024)
                text = data.decode(errors='ignore').strip()
                if not text:
                    continue
                self.response = text
                print(f"<- Received: {text}")
                lower = text.lower()
                if 'motor stop' in lower:
                    self.motor_fault = True
                    self.in_flight = False
                    self.last_motor_fault_time = time.time()
                    self.last_error = text
                    print("⚠ Motor fault detected: marking drone as not flying.")
                if 'no valid imu' in lower:
                    self.imu_fault = True
                    self.in_flight = False
                    self.last_error = text
                    print("⚠ IMU fault detected.")
            except socket.timeout:
                continue
            except Exception as e:
                print(f"❌ Receive thread error: {e}")
                break

    def send_command(self, command, timeout=7.0, retries=1):
        """
        Sends command; returns (ok: bool, resp: str|None).
        Query commands (ending with '?') return the raw string response.
        """
        if not command:
            return True, None

        cmd_root = command.split()[0]
        is_query = command.strip().endswith('?')

        if self.motor_fault and cmd_root not in ('land', 'battery?'):
            since = time.time() - self.last_motor_fault_time
            if since < 6.0:
                print(f"⚠ Suppressed movement command '{command}' due to recent motor fault ({since:.1f}s)")
                return False, "Motor Fault"
            else:
                print("Clearing stale motor fault.")
                self.motor_fault = False

        for i in range(retries + 1):
            print(f"-> Sending: {command}")
            try:
                self.sock.sendto(command.encode(), self.tello_address)
            except Exception as e:
                print(f"❌ UDP send failed: {e}")
                return False, f"UDP send failed: {e}"

            start = time.time()
            while time.time() - start < timeout:
                if self.response is not None:
                    text = self.response
                    self.response = None

                    if is_query and text:
                        return True, text

                    if text == 'ok':
                        if cmd_root == 'takeoff':
                            self.in_flight = True
                            print("✅ Tello state: in_flight = True")
                            time.sleep(4)
                        if cmd_root == 'land':
                            self.in_flight = False
                            print("✅ Tello state: in_flight = False")
                            time.sleep(3)
                        return True, "ok"
                    elif 'error' in text.lower():
                        self.last_error = text
                        if 'not joystick' in text.lower() and i < retries:
                            print("⚠ 'Not joystick' error. Re-entering SDK mode and retrying command.")
                            self.send_command('command')
                            time.sleep(1.0)
                            break
                        return False, text
                time.sleep(0.05)

            if i < retries:
                print(f"⚠ Timeout or error on attempt {i+1}. Retrying...")
                time.sleep(0.5)

        print(f"⚠ Command '{command}' failed after all retries.")
        return False, self.last_error or "Timeout"

    def safe_takeoff(self):
        if not self.in_flight:
            ok, _ = self.send_command("takeoff")
            return ok
        return False

    def safe_land(self):
        if self.in_flight:
            ok, _ = self.send_command("land")
            return ok
        return False

    def start_sdk_mode(self):
        ok, _ = self.send_command('command')
        if not ok:
            print("❌ Failed to enter SDK mode")
            return False

        ok, battery_level = self.send_command('battery?', timeout=3.5)
        if ok and battery_level and battery_level.strip().isdigit():
            battery = int(battery_level.strip())
            print(f"🔋 Battery level: {battery}%")
            if battery < MIN_BATTERY_LEVEL:
                print(f"🚨 LOW BATTERY WARNING: {battery}% (min {MIN_BATTERY_LEVEL}%).")
        else:
            print("⚠ Could not retrieve battery level.")

        # slow speed for mapping
        self.send_command(f"speed {TELLO_SPEED_CM_S}")

        # keep the drone stream on (ORB-SLAM consumes it separately)
        self.send_command('streamoff')
        time.sleep(0.3)
        ok, _ = self.send_command('streamon')
        if not ok:
            print("❌ Failed to enable video stream")
            return False
        print("✅ SDK mode active and video stream is on.")
        time.sleep(1.5)
        return True

    def stop(self):
        print("Stopping Tello controller...")
        self.stop_event.set()
        if self.response_thread.is_alive():
            self.response_thread.join(timeout=1.0)
        self.sock.close()
        print("Tello controller stopped.")

    def safe_rotate_cw(self, angle):
        ok, _ = self.send_command(f"cw {angle}", retries=ROTATION_RETRY_ATTEMPTS)
        if ok:
            time.sleep(1.8)
        return ok


# ---------------- ORB-SLAM monitor ----------------
class SlamCoverageMonitor:
    def __init__(self, logfile, sector_size_deg=30, min_samples_per_sector=2, min_sectors=10):
        self.logfile = logfile
        self.sector_size = sector_size_deg
        self.min_samples_per_sector = min_samples_per_sector
        self.min_sectors = min_sectors

        self.last_yaw_deg = None
        self.sector_counts = defaultdict(int)
        self.is_scanning = False

        self._stop = threading.Event()
        self._thread = None
        self._last_len = 0

    @staticmethod
    def quat_to_yaw_deg(qx, qy, qz, qw):
        r = R.from_quat([qx, qy, qz, qw])
        return r.as_euler('xyz', degrees=True)[2]

    def _sector_index(self, yaw_deg):
        y = (yaw_deg % 360 + 360) % 360
        return int(y // self.sector_size)

    def _run(self):
        print(f"🧭 [SLAM] Monitor started: {self.logfile}")
        while not self._stop.is_set():
            try:
                if not os.path.exists(self.logfile):
                    time.sleep(0.5); continue
                with open(self.logfile, 'r') as f:
                    lines = f.readlines()
                if len(lines) > self._last_len:
                    new_lines = lines[self._last_len:]
                    for line in new_lines:
                        parts = line.strip().split()
                        if len(parts) == 8:
                            qx, qy, qz, qw = map(float, parts[4:])
                            yaw = self.quat_to_yaw_deg(qx, qy, qz, qw)
                            self.last_yaw_deg = yaw
                            if self.is_scanning:
                                idx = self._sector_index(yaw)
                                self.sector_counts[idx] += 1
                    self._last_len = len(lines)
            except Exception as e:
                print(f"❌ [SLAM] Monitor error: {e}")
            time.sleep(0.2)

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.0)

    def clear_log(self):
        try:
            if os.path.exists(self.logfile):
                open(self.logfile, 'w').close()
            self._last_len = 0
            self.sector_counts.clear()
            print("🧭 [SLAM] Log cleared for new scan.")
        except Exception as e:
            print(f"❌ [SLAM] Could not clear log: {e}")

    def mapping_ready(self):
        covered = sum(1 for c in self.sector_counts.values() if c >= self.min_samples_per_sector)
        return covered >= self.min_sectors

    def sector_mapped(self, yaw_deg):
        idx = self._sector_index(yaw_deg if yaw_deg is not None else 0.0)
        return self.sector_counts.get(idx, 0) >= self.min_samples_per_sector


# ---------------- Controller (webcam gestures + SLAM gating) ----------------
class SlamFeedbackController:
    def __init__(self, tello_controller, gesture_cam_index=GESTURE_CAMERA_INDEX):
        self.tello = tello_controller
        self.gesture_cam_index = gesture_cam_index

        self.hands = mp_hands.Hands(
            static_image_mode=False, max_num_hands=1,
            model_complexity=0, min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.mp_draw = mp.solutions.drawing_utils

        self.GESTURE_MAP = {
            "palm":        "takeoff",
            "call_me":     "land",
            "thumbs_up":   f"forward {MOVEMENT_DISTANCE_CM}",
            "thumbs_down": f"back {MOVEMENT_DISTANCE_CM}",
            "peace":       f"cw {INCREMENTAL_ANGLE}",
            "okay":        f"ccw {INCREMENTAL_ANGLE}",
            "index_only":  f"up {ALTITUDE_STEP_CM}",
            "fist":        f"down {ALTITUDE_STEP_CM}",
            "stop":        None,
            "spiderman":   "cw 360",
        }

        self.gesture_history = deque(maxlen=DEBOUNCE_HISTORY)
        self.last_command_time = 0.0
        self.check360_hold = 0
        self.is_scanning_now = False
        self.initial_scan_done = False

        # SLAM monitor
        self.slam = SlamCoverageMonitor(
            ORB_SLAM_LOG_FILE,
            sector_size_deg=SECTOR_SIZE_DEG,
            min_samples_per_sector=MAPPING_MIN_SAMPLES_PER_SECTOR,
            min_sectors=MAPPING_MIN_SECTORS
        )
        self.slam.start()

    # ---------- SLAM-driven scans & gating ----------
    def _scan_360_slow(self, announce="🔁 Starting slow 360° scan for mapping..."):
        if not self.tello.in_flight or self.tello.motor_fault:
            print("⚠ Cannot scan: not in flight or fault active.")
            return
        print(announce)
        self.is_scanning_now = True
        self.slam.is_scanning = True
        self.slam.clear_log()
        steps = 360 // INCREMENTAL_ANGLE
        for i in range(steps):
            if not self.tello.safe_rotate_cw(INCREMENTAL_ANGLE):
                print(f"⚠ Rotation step {i+1} failed. Aborting 360.")
                break
            time.sleep(INCREMENTAL_PAUSE)
        self.slam.is_scanning = False
        self.is_scanning_now = False
        covered = sum(1 for c in self.slam.sector_counts.values() if c >= MAPPING_MIN_SAMPLES_PER_SECTOR)
        print(f"🔁 Scan done. Sectors covered: {covered}/12")

    def _mapped_direction_ok(self, direction):
        # If gating is off, always allow
        if not SLAM_GATING_ENABLED:
            return True
        yaw = self.slam.last_yaw_deg
        if yaw is None:
            print("⚠ SLAM yaw unknown; blocking movement.")
            return False
        check_yaw = yaw if direction == 'forward' else (yaw + 180.0)
        if self.slam.sector_mapped(check_yaw):
            return True
        print(f"⚠ Target sector ({direction}) not mapped by SLAM; blocking movement.")
        return False

    # ---------- main loop ----------
    def run(self):
        if not self.tello.start_sdk_mode():
            self.stop()
            return

        # Open webcam for gestures
        cap = cv2.VideoCapture(self.gesture_cam_index)
        if not cap.isOpened():
            print(f"❌ Could not open webcam at index {self.gesture_cam_index}. Exiting.")
            self.stop()
            return

        ok, first_frame = cap.read()
        if not ok or first_frame is None:
            print("❌ Could not read initial frame from webcam. Exiting.")
            cap.release()
            self.stop()
            return

        h, w = first_frame.shape[:2]
        print("🎯 Gesture controller running (WEBCAM). Drone stream is for SLAM only.")
        emoji_map = {"palm":"🖐", "call_me":"🤙", "thumbs_up":"👍", "thumbs_down":"👎",
                     "peace":"✌", "okay":"👌", "index_only":"☝", "fist":"✊",
                     "stop":"✋", "spiderman":"🤟"}

        while True:
            success, image = cap.read()
            if not success or image is None:
                print("⚠ Webcam frame grab failed. Exiting.")
                break

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.hands.process(image_rgb)
            current_gesture = None
            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                self.mp_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                current_gesture = get_gesture(hand_landmarks)

            self.gesture_history.append(current_gesture)
            stable_gesture = None
            if len(self.gesture_history) == DEBOUNCE_HISTORY:
                last_gest = self.gesture_history[-1]
                if last_gest is not None and list(self.gesture_history).count(last_gest) >= STEADY_REQUIRED:
                    stable_gesture = last_gest

            display_gesture = stable_gesture or current_gesture
            if display_gesture:
                emoji = emoji_map.get(display_gesture, "❓")
                cv2.putText(image, f"{emoji} {display_gesture}", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2, cv2.LINE_AA)

            # HUD: SLAM status (still shown, but won't block fwd/back if gating off)
            yaw_txt = "?" if self.slam.last_yaw_deg is None else f"{self.slam.last_yaw_deg:.1f}°"
            mapped_sectors = sum(1 for c in self.slam.sector_counts.values() if c >= MAPPING_MIN_SAMPLES_PER_SECTOR)
            ready_txt = "(OK)" if self.slam.mapping_ready() else "(scanning)"
            if not SLAM_GATING_ENABLED:
                ready_txt = "(gating off)"
            map_txt = f"SLAM sectors: {mapped_sectors}/12 {ready_txt}"
            cv2.putText(image, f"Yaw:{yaw_txt}  {map_txt}", (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

            now = time.time()
            cooldown_passed = (now - self.last_command_time > COMMAND_COOLDOWN)

            # hold spiderman to manual scan
            if stable_gesture == 'spiderman':
                self.check360_hold += 1
            else:
                self.check360_hold = 0

            if self.check360_hold >= CHECK360_HOLD_REQUIRED and not self.is_scanning_now:
                self._scan_360_slow(announce="🤟 Manual 360° scan (spiderman) ...")
                self.check360_hold = 0
                self.last_command_time = now

            elif stable_gesture and cooldown_passed and not self.is_scanning_now:
                if self.tello.motor_fault and stable_gesture != 'call_me':
                    print("⚠ Drone fault active. Only LAND (call me 🤙) is allowed.")
                    self.last_command_time = now
                    continue

                if stable_gesture == 'palm' and not self.tello.in_flight:
                    print("🖐 Palm detected -> TAKEOFF")
                    if self.tello.safe_takeoff():
                        time.sleep(1.0)
                        # optional initial scan (kept)
                        self._scan_360_slow()
                        self.initial_scan_done = True
                    self.last_command_time = now

                elif stable_gesture == 'call_me' and self.tello.in_flight:
                    print("🤙 Call Me detected -> LAND")
                    self.tello.safe_land()
                    self.last_command_time = now

                else:
                    command = self.GESTURE_MAP.get(stable_gesture)
                    if command and self.tello.in_flight:
                        root = command.split()[0]
                        if root in ('forward', 'back'):
                            # With gating off, always allow
                            if not self._mapped_direction_ok('forward' if root=='forward' else 'back'):
                                self.last_command_time = now
                                continue
                        print(f"{emoji_map.get(stable_gesture,'')} {stable_gesture} -> Sending: {command}")
                        self.tello.send_command(command)
                        self.last_command_time = now
                    elif stable_gesture == 'stop':
                        print("✋ Stop -> Hovering (no-op)")

            status = f"Flight:{self.tello.in_flight} | MotorFault:{self.tello.motor_fault}"
            status_color = (0, 0, 255) if self.tello.motor_fault else (255, 255, 0)
            cv2.putText(image, status, (20, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

            cv2.imshow("Gesture Controller (Webcam + SLAM gating)", image)
            if cv2.waitKey(1) & 0xFF == 27:
                break

        print("Exiting...")
        if self.tello.in_flight:
            print("Attempting to land before closing...")
            self.tello.safe_land()
        self.stop()
        if cap:
            cap.release()
        cv2.destroyAllWindows()

    def stop(self):
        self.tello.stop()
        self.hands.close()
        self.slam.stop()


# -------------- main --------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--webcam-index", type=int, default=GESTURE_CAMERA_INDEX,
                        help="OpenCV index for the webcam used for gesture recognition.")
    args = parser.parse_args()

    controller = None
    try:
        tello = TelloController()
        controller = SlamFeedbackController(tello, gesture_cam_index=args.webcam_index)
        controller.run()
    except Exception as e:
        print(f"An unhandled error occurred in main: {e}")
    finally:
        if controller:
            controller.stop()