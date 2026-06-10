import time
import sys
import threading
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

# =====================================================================
#  main_freespace.py -- Free-Space Navigation (BFR pseudo code)
#
#  TRIET LY DUNG:
#    Column scan quet TU DAY LEN, dung o edge dau tien (chan vat can).
#    y_local = chieu cao tu day ROI den chan vat can = KHOANG TRONG.
#    => c CAO = THOANG | c THAP = bi chan.
#    => Di ve huong co c LON NHAT (max) = thoang nhat.
#
#  3 TRANG THAI:
#    CRUISE   : di ve huong thoang nhat. F cao -> di thang.
#    AVOIDING : phia truoc bi chan -> khoa 1 huong xoay den khi thoang.
#    STUCK    : ca 3 vung ~0 HOAC ca 3 dao dong +-5px suot 10s (ket)
#               -> lui 2s -> re trai 2s -> tim duong tiep.
# =====================================================================

# --- CAU HINH chung ---
CAM_INDEX     = 0
FRAME_W       = 320
FRAME_H       = 240
DISPLAY_SCALE = 2

# --- Canny + column scan ---
STEP_SIZE     = 8
CANNY_LOW     = 50
CANNY_HIGH    = 100
ROI_TOP_RATIO = 0.35
SAFE_BOTTOM   = 5

# --- Free-space decision ---
BLOCK_RATIO     = 0.30   # c_fwd < BLOCK_RATIO*roi_h => phia truoc bi chan
STEER_MARGIN_PX = 15     # L/R phai thoang hon F it nhat ngan nay moi re
N_REGIONS       = 3

# --- STUCK detection ---
STUCK_ZERO_SUM = 1.0     # tong 3 vung < nguong -> all-zero
STUCK_WINDOW_S = 10.0    # cua so theo doi "ket"
STUCK_OSC_BAND = 10.0    # chi can IT NHAT 1 VUNG dao dong trong +-10px suot window -> ket
                         # (khong can ca 3 cung yem tinh)
STUCK_BACK_S   = 2.5   # lui 4s
STUCK_TURN_S   = 3.0  # re trai toi khi thay thoang duong (khong timeout)

# --- FSM ---
CONFIRM_FRAMES = 2
EXIT_CONFIRM   = 3
WARM_UP_FRAMES = 2

# --- Timing ---
ACT_DURATION  = 0.3
TURN_NUDGE_S  = 0.3
ENTRY_BACK_S  = 0.35

# --- Hardware ---
USE_ARDUINO  = True
ARDUINO_PORT = '/dev/ttyACM0'
ARDUINO_BAUD = 9600

# --- Save & stream ---
SAVE_VIDEO    = True
OUTPUT_PATH   = 'run_freespace.avi'
STREAM_ENABLE = True
STREAM_PORT   = 8000


# ---------------------------------------------------------------------
#  MJPEG STREAM
# ---------------------------------------------------------------------
class FrameBuffer:
    def __init__(self):
        self._jpeg = None
        self._lock = threading.Lock()
    def update(self, bgr_image):
        ok, buf = cv2.imencode('.jpg', bgr_image, [cv2.IMWRITE_JPEG_QUALITY, 75])
        if ok:
            with self._lock: self._jpeg = buf.tobytes()
    def get(self):
        with self._lock: return self._jpeg

_frame_buffer = FrameBuffer()

class _MJPEGHandler(BaseHTTPRequestHandler):
    def log_message(self, *args): pass
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
        self.end_headers()
        try:
            while True:
                jpeg = _frame_buffer.get()
                if jpeg is None:
                    time.sleep(0.05); continue
                self.wfile.write(b'--frame\r\n')
                self.send_header('Content-Type', 'image/jpeg')
                self.send_header('Content-Length', str(len(jpeg)))
                self.end_headers()
                self.wfile.write(jpeg); self.wfile.write(b'\r\n')
                time.sleep(0.1)
        except (BrokenPipeError, ConnectionResetError): pass

def start_stream_server(port):
    server = ThreadingHTTPServer(('0.0.0.0', port), _MJPEGHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


# ---------------------------------------------------------------------
#  MOTOR CONTROLLER
# ---------------------------------------------------------------------
class MotorController:
    def __init__(self, use_arduino, port, baud):
        self.ser = None
        self.last_cmd = None
        if use_arduino:
            try:
                import serial
                self.ser = serial.Serial(port, baud, timeout=1)
                time.sleep(2)
                print("[motor] Arduino CONNECTED:", port)
            except Exception as e:
                print("[motor] CONNECTION FAIL:", e)
        else:
            print("[motor] SIMULATION MODE (USE_ARDUINO=False)")

    def send(self, cmd):
        if cmd == self.last_cmd: return
        print("[motor] ->", cmd)
        self.last_cmd = cmd
        if self.ser is not None:
            try: self.ser.write(cmd.encode())
            except Exception as e: print("[motor] SERIAL ERROR:", e)

    def stop(self): self.send('x')

    def close(self):
        if self.ser is not None:
            self.stop()
            self.ser.close()


# ---------------------------------------------------------------------
#  CANNY + COLUMN SCAN
# ---------------------------------------------------------------------
def scan_edges(frame_bgr):
    """
    Quet tu day len, tim edge dau tien (chan vat can).
    y_local = khoang trong tu day ROI den chan vat can.
    c CAO = THOANG | c THAP = bi chan.
    """
    h, w = frame_bgr.shape[:2]

    roi_top = int(h * ROI_TOP_RATIO)
    roi = frame_bgr[roi_top:h, :]
    roi_h, roi_w = roi.shape[:2]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 9, 30, 30)
    edges = cv2.Canny(gray, CANNY_LOW, CANNY_HIGH)

    edge_array = []
    for x in range(0, roi_w, STEP_SIZE):
        found = False
        for y in range(roi_h - SAFE_BOTTOM, 0, -1):
            if edges[y, x] == 255:
                y_local = roi_h - 1 - y    # khoang trong tu day
                edge_array.append((x, y_local))
                found = True
                break
        if not found:
            edge_array.append((x, roi_h - 1))   # khong edge -> thoang het tam

    n = len(edge_array)
    if n < N_REGIONS:
        c_left = c_fwd = c_right = 0.0
    else:
        chunk = n // N_REGIONS
        left_ys  = [p[1] for p in edge_array[0:chunk]]
        fwd_ys   = [p[1] for p in edge_array[chunk:2*chunk]]
        right_ys = [p[1] for p in edge_array[2*chunk:]]
        c_left  = float(np.mean(left_ys))  if left_ys  else 0.0
        c_fwd   = float(np.mean(fwd_ys))   if fwd_ys   else 0.0
        c_right = float(np.mean(right_ys)) if right_ys else 0.0

    return {
        'edge_array': edge_array,
        'roi_h':      roi_h,
        'roi_w':      roi_w,
        'roi_top':    roi_top,
        'edges':      edges,
        'c_left':     c_left,
        'c_fwd':      c_fwd,
        'c_right':    c_right,
    }


# ---------------------------------------------------------------------
#  STUCK TRACKER
# ---------------------------------------------------------------------
class StuckTracker:
    """
    Phat hien ket:
      1. ALL-ZERO  : (c_l + c_f + c_r) < STUCK_ZERO_SUM
      2. OSCILLATE : ca 3 vung dao dong trong +-STUCK_OSC_BAND
                     suot STUCK_WINDOW_S giay (so do gan nhu khong doi).
    """
    def __init__(self):
        self.history = deque()   # (timestamp, c_l, c_f, c_r)

    def reset(self):
        self.history.clear()

    def update(self, c_l, c_f, c_r):
        now = time.time()
        self.history.append((now, c_l, c_f, c_r))
        while self.history and now - self.history[0][0] > STUCK_WINDOW_S:
            self.history.popleft()

    def is_all_zero(self, c_l, c_f, c_r):
        return (c_l + c_f + c_r) < STUCK_ZERO_SUM

    def is_oscillating(self):
        """
        Chi can IT NHAT 1 vung dao dong trong +-STUCK_OSC_BAND suot STUCK_WINDOW_S.
        Khi xe bi ket, thuong 1-2 vung yem, khong phai ca 3 cung yem.
        """
        if not self.history:
            return False
        span = self.history[-1][0] - self.history[0][0]
        if span < STUCK_WINDOW_S * 0.9:
            return False
        ls = [s[1] for s in self.history]
        fs = [s[2] for s in self.history]
        rs = [s[3] for s in self.history]
        def amp(v): return max(v) - min(v)
        # OR logic: neu chi can 1 vung yem (dao dong < band) -> coi la ket
        return (amp(ls) <= STUCK_OSC_BAND or
                amp(fs) <= STUCK_OSC_BAND or
                amp(rs) <= STUCK_OSC_BAND)


# ---------------------------------------------------------------------
#  FSM (CRUISE / AVOIDING / STUCK)
# ---------------------------------------------------------------------
class FreeSpaceFSM:
    CRUISE   = "CRUISE"
    AVOIDING = "AVOIDING"
    STUCK    = "STUCK"

    def __init__(self):
        self.state = self.CRUISE
        self.locked_turn_dir = None
        self.danger_streak   = 0
        self.clear_streak    = 0
        self.just_entered    = False
        self.warm_up         = 0
        self.stuck_phase     = None
        self.stuck_t0        = None
        self.stuck_detected_time = None  # thoi gian phat hien stuck (de hien countdown)

    def _blocked_front(self, info, roi_h):
        return info['c_fwd'] < BLOCK_RATIO * roi_h

    def _front_clear(self, info, roi_h):
        return info['c_fwd'] >= BLOCK_RATIO * roi_h * 1.4

    def _most_open_turn(self, info):
        return 'a' if info['c_left'] >= info['c_right'] else 'd'

    def _enter_stuck(self, tag):
        self.state = self.STUCK
        self.stuck_phase = None
        self.stuck_t0 = None
        self.stuck_detected_time = time.time()  # ghi lai thoi diem phat hien
        self.danger_streak = 0
        self.clear_streak = 0
        return ('x', "ENTER STUCK (%s): back 4s + turn until clear" % tag, self.state)

    def step(self, info, roi_h, stuck_flag):
        c_l, c_f, c_r = info['c_left'], info['c_fwd'], info['c_right']

        # ===== STUCK escape =====
        if self.state == self.STUCK:
            if self.stuck_phase is None:
                self.stuck_phase = 'backingup'
                self.stuck_t0 = time.time()
                return ('s', "STUCK: backing up (4s)", self.state)
            elapsed = time.time() - self.stuck_t0
            if self.stuck_phase == 'backingup':
                if elapsed < STUCK_BACK_S:
                    return ('s', "STUCK: backing %.1f/%.1fs" % (elapsed, STUCK_BACK_S), self.state)
                self.stuck_phase = 'turning'
                self.stuck_t0 = time.time()
                return ('a', "STUCK: turning left (until clear)", self.state)
            
            # turning phase: re trai toi khi thay thoang duong (c_fwd cao)
            if self._front_clear(info, roi_h):
                # thoang roi, thoat STUCK
                self.state = self.CRUISE
                self.stuck_phase = None
                self.stuck_t0 = None
                self.stuck_detected_time = None
                self.warm_up = WARM_UP_FRAMES
                return ('w', "STUCK: found clear -> CRUISE", self.state)
            # chua thoang -> tiep tuc re trai
            return ('a', "STUCK: turning left (F=%.0f)" % c_f, self.state)

        # ===== CRUISE =====
        if self.state == self.CRUISE:
            if stuck_flag:
                return self._enter_stuck("all-zero/oscillate")
            if self.warm_up > 0:
                self.warm_up -= 1
                self.danger_streak = 0
                return ('w', "WARM-UP %d/%d (F=%.0f)" % (self.warm_up, WARM_UP_FRAMES, c_f), self.state)

            if self._blocked_front(info, roi_h):
                self.danger_streak += 1
                if self.danger_streak >= CONFIRM_FRAMES:
                    self.locked_turn_dir = self._most_open_turn(info)
                    self.state = self.AVOIDING
                    self.just_entered = True
                    self.clear_streak = 0
                    self.danger_streak = 0
                    return ('x', "ENTER AVOIDING, lock=%s (L=%.0f F=%.0f R=%.0f)" %
                            (self.locked_turn_dir, c_l, c_f, c_r), self.state)
                turn = self._most_open_turn(info)
                return (turn, "pre-avoid: nudge %s (L=%.0f F=%.0f R=%.0f)" %
                        (turn, c_l, c_f, c_r), self.state)

            self.danger_streak = 0
            if c_l > c_f + STEER_MARGIN_PX and c_l >= c_r:
                return ('a', "CRUISE: lean left (L=%.0f F=%.0f R=%.0f)" % (c_l, c_f, c_r), self.state)
            if c_r > c_f + STEER_MARGIN_PX and c_r > c_l:
                return ('d', "CRUISE: lean right (L=%.0f F=%.0f R=%.0f)" % (c_l, c_f, c_r), self.state)
            return ('w', "CRUISE: forward (L=%.0f F=%.0f R=%.0f)" % (c_l, c_f, c_r), self.state)

        # ===== AVOIDING =====
        if self.state == self.AVOIDING:
            if stuck_flag:
                return self._enter_stuck("stuck-in-avoiding")
            if self.just_entered:
                self.just_entered = False
                return ('s', "AVOIDING: entry back-up", self.state)
            if self._front_clear(info, roi_h):
                self.clear_streak += 1
                if self.clear_streak >= EXIT_CONFIRM:
                    old = self.locked_turn_dir
                    self.state = self.CRUISE
                    self.locked_turn_dir = None
                    self.clear_streak = 0
                    self.warm_up = WARM_UP_FRAMES
                    return ('w', "EXIT AVOIDING (was=%s) -> CRUISE" % old, self.state)
                return ('w', "clear streak %d/%d (F=%.0f)" %
                        (self.clear_streak, EXIT_CONFIRM, c_f), self.state)
            self.clear_streak = 0
            return (self.locked_turn_dir,
                    "AVOIDING: turning %s (L=%.0f F=%.0f R=%.0f)" %
                    (self.locked_turn_dir, c_l, c_f, c_r), self.state)

        return ('w', "DEFAULT", self.state)


# ---------------------------------------------------------------------
#  BUILD VIEW
# ---------------------------------------------------------------------
def build_view(frame_bgr, info, cmd, reason, state, fps, stuck_flag, stuck_detected_time=None):
    h, w = frame_bgr.shape[:2]
    original = frame_bgr.copy()
    result = frame_bgr.copy()

    roi_top = info['roi_top']
    roi_h   = info['roi_h']

    edge_array = info['edge_array']
    pts_img = []
    for (x, y_local) in edge_array:
        y_img = (roi_top + roi_h - 1) - y_local
        pts_img.append((x, y_img))

    for i in range(len(pts_img) - 1):
        cv2.line(result, pts_img[i], pts_img[i + 1], (0, 255, 0), 2)
    bottom_y = roi_top + roi_h - 1
    for i in range(0, len(pts_img), 2):
        cv2.line(result, (pts_img[i][0], bottom_y), pts_img[i], (0, 180, 0), 1)

    region_w = w // N_REGIONS
    for i in range(1, N_REGIONS):
        x = i * region_w
        cv2.line(result, (x, roi_top), (x, h), (255, 200, 0), 1)

    block_y_local = int(BLOCK_RATIO * roi_h)
    block_y_img = (roi_top + roi_h - 1) - block_y_local
    cv2.line(result, (0, block_y_img), (w, block_y_img), (0, 0, 255), 1)
    cv2.putText(result, "BLOCK line", (w - 110, block_y_img - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    cs = [info['c_left'], info['c_fwd'], info['c_right']]
    labels = ['L', 'F', 'R']
    for i, (label, c) in enumerate(zip(labels, cs)):
        x0 = i * region_w + region_w // 2 - 20
        color = (0, 220, 0) if c >= BLOCK_RATIO * roi_h else (0, 0, 255)
        cv2.putText(result, "%s=%.0f" % (label, c), (x0, h - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    _label(original, "VIDEO")
    _label(result, "RESULT (free-space skyline)")

    info_h = 90
    canvas = np.zeros((h + info_h, w * 2, 3), dtype=np.uint8)
    canvas[0:h, 0:w] = original
    canvas[0:h, w:w * 2] = result

    state_color = {'CRUISE': (0, 220, 0),
                   'AVOIDING': (0, 100, 255),
                   'STUCK': (255, 100, 0)}.get(state, (255, 255, 255))
    cmd_color = {'w': (0, 220, 0), 'x': (0, 0, 255),
                 'a': (0, 200, 255), 'd': (0, 200, 255),
                 's': (200, 200, 0)}.get(cmd, (255, 255, 255))

    cv2.putText(canvas, "STATE: %s" % state, (12, h + 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, state_color, 2)
    cv2.putText(canvas, "CMD: %s" % cmd, (260, h + 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, cmd_color, 2)
    cv2.putText(canvas, reason, (12, h + 56),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)
    
    # Hien thi STUCK resolve countdown neu dang trong STUCK state
    stuck_resolve_txt = ""
    if stuck_detected_time is not None:
        elapsed = time.time() - stuck_detected_time
        if elapsed <= 10.0:
            stuck_resolve_txt = "STUCK resolve: %.1f/10s" % elapsed
    
    cv2.putText(canvas, "FPS: %.1f  %s" % (fps, stuck_resolve_txt),
                (w * 2 - 300, h + 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 100, 255) if stuck_resolve_txt else (180, 180, 180), 1)
    cv2.putText(canvas, "c HIGH = open | go to max(c) | stuck -> back+turn",
                (12, h + 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (140, 140, 140), 1)

    if DISPLAY_SCALE != 1:
        canvas = cv2.resize(canvas,
                            (canvas.shape[1] * DISPLAY_SCALE, canvas.shape[0] * DISPLAY_SCALE),
                            interpolation=cv2.INTER_LINEAR)
    return canvas

def _label(img, text):
    cv2.rectangle(img, (0, 0), (img.shape[1], 22), (40, 40, 40), -1)
    cv2.putText(img, text, (8, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


# ---------------------------------------------------------------------
#  MAIN LOOP
# ---------------------------------------------------------------------
def main():
    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    if not cap.isOpened():
        print("[ERROR] Cannot open camera"); sys.exit(1)

    print("[config] Free-space: c HIGH=open, go to max(c)")
    print("[config] BLOCK_RATIO=%.2f STEER_MARGIN=%dpx" % (BLOCK_RATIO, STEER_MARGIN_PX))
    print("[config] STUCK: all-zero OR osc +-%.0fpx for %.0fs -> back %.0fs + turn until clear" %
          (STUCK_OSC_BAND, STUCK_WINDOW_S, STUCK_BACK_S))

    motor   = MotorController(USE_ARDUINO, ARDUINO_PORT, ARDUINO_BAUD)
    fsm     = FreeSpaceFSM()
    tracker = StuckTracker()

    writer = None
    if SAVE_VIDEO:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, 10.0,
                                 (FRAME_W * 2, FRAME_H + 90))

    if STREAM_ENABLE:
        start_stream_server(STREAM_PORT)
        print("[stream] http://<pi-ip>:%d" % STREAM_PORT)

    print("[run] STARTED -- Ctrl+C to stop")

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("[ERROR] read fail"); break

            t0 = time.time()
            info = scan_edges(frame)
            c_l, c_f, c_r = info['c_left'], info['c_fwd'], info['c_right']

            tracker.update(c_l, c_f, c_r)
            stuck_flag = (tracker.is_all_zero(c_l, c_f, c_r) or
                          tracker.is_oscillating())

            cmd, reason, state = fsm.step(info, info['roi_h'], stuck_flag)

            # khi vao STUCK thi reset tracker de khong re-trigger lien tuc
            if state == FreeSpaceFSM.STUCK and fsm.stuck_phase == 'backingup':
                tracker.reset()

            proc_time = time.time() - t0
            fps = 1.0 / max(proc_time, 1e-6)

            view = build_view(frame, info, cmd, reason, state, fps, stuck_flag, fsm.stuck_detected_time)
            if STREAM_ENABLE: _frame_buffer.update(view)
            if writer is not None: writer.write(view)

            motor.send(cmd)
            if cmd == 's':
                time.sleep(ENTRY_BACK_S)
            elif state == "AVOIDING" and cmd in ('a', 'd'):
                time.sleep(TURN_NUDGE_S)
            else:
                time.sleep(ACT_DURATION)

    except KeyboardInterrupt:
        print("\n[run] STOP")
    finally:
        motor.close()
        cap.release()
        if writer is not None:
            writer.release()
            print("[run] Video saved:", OUTPUT_PATH)
        print("[run] FINISH")


if __name__ == "__main__":
    main()
