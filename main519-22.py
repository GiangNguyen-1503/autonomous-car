import time
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

# =====================================================================
#  main519_fsm.py  -- FSM 2 trang thai CRUISE / AVOIDING
#  Sua loi "lui-xoay loan": xe se chon mot huong duy nhat khi vao trang
#  thai vuot can va GIU huong do cho den khi tim duoc loi thoang.
# =====================================================================

# --- CAU HINH chung ---
CAM_INDEX   = 0
FRAME_W     = 320
FRAME_H     = 240
DISPLAY_SCALE = 2

# --- Auto-calibration ---
AUTO_CALIBRATE = True
CALIB_SECONDS  = 3.0

# --- Floor detection (fallback values, se duoc auto-calibrate ghi de) ---
FLOOR_THRESH = 40
FREE_MIN     = 0.45
STEER_MARGIN = 0.08
N_COLS       = 3

# --- DANGER detection ---
DANGER_LINE_Y = 0.70
DANGER_MIN    = 0.30    # [TANG len 0.30]: it nhay hon, chi trigger khi vat thuc su rat gan
DANGER_SAFE   = 0.45    # Trong AVOIDING, can danger_ratio >= muc nay moi exit

# --- FSM timing & confirm ---
ACT_DURATION    = 0.3      # nhip vong lap thong thuong
CONFIRM_FRAMES  = 4        # [TANG len 4]: can nhieu khung DANGER hon moi chuyen CRUISE -> AVOIDING
CONFIRM_PAUSE_FRAMES = 3   # [MOI]: so khung doc them de confirm truoc khi doi lenh
EXIT_CONFIRM    = 3        # so khung "thoang" lien tiep moi thoat AVOIDING
ENTRY_BACK_S    = 0.4      # lui 1 lan luc VAO AVOIDING (de co khoang xoay)
TURN_NUDGE_S    = 0.3      # moi nhip xoay trong AVOIDING
WARM_UP_FRAMES  = 2        # sau khi exit AVOIDING, khong trigger DANGER trong N khung

# --- Hardware ---
USE_ARDUINO  = True
ARDUINO_PORT = '/dev/ttyACM0'
ARDUINO_BAUD = 9600

# --- Save & stream ---
SAVE_VIDEO    = True
OUTPUT_PATH   = 'run_output.avi'
STREAM_ENABLE = True
STREAM_PORT   = 8000

# --- Calibration sanity check & poisoned detection ---
CALIB_MIN_RATIO = 0.20      # Giam tu 0.50 -> 0.20: cho phep calibrate o moi truong
                            # co vat can trong tam nhin (chap nhan noise)
POISON_STD_K    = 3.0       # Tang tu 2.0 -> 3.0: it nhay hon (99.7% thay vi 95%)

# --- Thresholds dong ---
THRESHOLDS = {
    'floor': FLOOR_THRESH,
    'free':  FREE_MIN,
    'danger': DANGER_MIN,
    'floor_hist': None,
    'poison_threshold': 0.0,
}


# ---------------------------------------------------------------------
#  AUTO-CALIBRATE  (giu nguyen logic, chi doi nguong sanity check)
# ---------------------------------------------------------------------
def calibrate_thresholds(cap, seconds=CALIB_SECONDS):
    print("[calibrate] START CALIBRATING (3s, dat xe truoc san trong)")
    for i in range(3, 0, -1):
        print("[calibrate] %d..." % i)
        time.sleep(1)
    print("[calibrate] MEASURING...")

    hist_samples = []
    sample_bp_means = []
    ratios_samples = []
    danger_samples = []
    otsu_samples = []

    t_end = time.time() + seconds
    while time.time() < t_end:
        ok, frame = cap.read()
        if not ok or frame is None:
            continue

        h, w = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # CAI TIEN: bilateral filter giong nhu detect_floor (de nguong khop)
        hsv = cv2.bilateralFilter(hsv, 5, 50, 50)
        sx1, sx2 = int(w * 0.30), int(w * 0.70)
        sy1, sy2 = int(h * 0.85), h
        floor = hsv[sy1:sy2, sx1:sx2]
        hist = cv2.calcHist([floor], [0, 1], None, [30, 32], [0, 180, 0, 256])
        cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
        hist_samples.append(hist)

        ry1, ry2 = int(h * 0.50), sy1
        region = hsv[ry1:ry2, :]
        bp = cv2.calcBackProject([region], [0, 1], hist, [0, 180, 0, 256], scale=1)
        bp = cv2.medianBlur(bp, 5)

        otsu_thresh, bp_bin = cv2.threshold(bp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        otsu_samples.append(otsu_thresh)

        col_w = w // N_COLS
        r = []
        for i in range(N_COLS):
            col = bp_bin[:, i * col_w:(i + 1) * col_w]
            r.append((col > 0).sum() / max(col.size, 1))
        ratios_samples.append(r)

        danger_y = max(0, int(h * DANGER_LINE_Y) - ry1)
        if danger_y < bp_bin.shape[0]:
            near = bp_bin[danger_y:, col_w:2 * col_w]
            if near.size > 0:
                danger_samples.append((near > 0).sum() / near.size)

        sample_bp = cv2.calcBackProject([floor], [0, 1], hist, [0, 180, 0, 256], scale=1)
        sample_bp_means.append(float(np.mean(sample_bp)))

    if not otsu_samples:
        print("[calibrate] FAIL: no frames. Using fallback.")
        return False

    avg_ratio = float(np.mean([np.mean(r) for r in ratios_samples]))
    if avg_ratio < CALIB_MIN_RATIO:
        print("[calibrate] REJECT: avg_ratio %.2f < %.2f -> fallback" % (avg_ratio, CALIB_MIN_RATIO))
        return False

    new_floor  = max(15, min(120, int(round(float(np.median(otsu_samples))))))
    new_free   = max(0.20, min(0.80, round(avg_ratio * 0.7, 2)))
    new_danger = round(float(np.mean(danger_samples)) * 0.5, 2) if danger_samples else DANGER_MIN
    new_danger = max(0.10, min(0.60, new_danger))

    THRESHOLDS['floor']  = new_floor
    THRESHOLDS['free']   = new_free
    THRESHOLDS['danger'] = new_danger

    avg_hist = np.mean(np.stack(hist_samples), axis=0)
    cv2.normalize(avg_hist, avg_hist, 0, 255, cv2.NORM_MINMAX)
    THRESHOLDS['floor_hist'] = avg_hist

    bp_mean = float(np.mean(sample_bp_means))
    bp_std  = float(np.std(sample_bp_means))
    THRESHOLDS['poison_threshold'] = max(0.0, bp_mean - POISON_STD_K * bp_std)

    print("[calibrate] DONE: floor=%d free=%.2f danger=%.2f" % (new_floor, new_free, new_danger))
    print("[calibrate] bp_mean=%.1f std=%.1f poison_thresh=%.1f" % (bp_mean, bp_std, THRESHOLDS['poison_threshold']))
    return True


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
        self.prev_cmd = None   # [MOI]: luu lenh truoc do de phat hien doi lenh
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
        self.prev_cmd = self.last_cmd  # [MOI]: luu lenh cu truoc khi cap nhat
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
#  DETECT FLOOR
# ---------------------------------------------------------------------
def detect_floor(frame_bgr):
    h, w = frame_bgr.shape[:2]
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    # CAI TIEN: bilateral filter lam muot van san truoc khi xu ly.
    # Giu canh sac net (vat can) nhung lam muot tong mau (san co van).
    # Giam nhieu detection do van san khac to mau.
    hsv = cv2.bilateralFilter(hsv, 5, 50, 50)

    sx1, sx2 = int(w * 0.30), int(w * 0.70)
    sy1, sy2 = int(h * 0.85), h
    floor = hsv[sy1:sy2, sx1:sx2]

    if THRESHOLDS['floor_hist'] is not None:
        hist = THRESHOLDS['floor_hist']
    else:
        hist = cv2.calcHist([floor], [0, 1], None, [30, 32], [0, 180, 0, 256])
        cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)

    rx1, rx2 = 0, w
    ry1, ry2 = int(h * 0.50), sy1
    region = hsv[ry1:ry2, rx1:rx2]
    bp = cv2.calcBackProject([region], [0, 1], hist, [0, 180, 0, 256], scale=1)
    bp = cv2.medianBlur(bp, 5)

    _, bp_bin = cv2.threshold(bp, THRESHOLDS['floor'], 255, cv2.THRESH_BINARY)

    mask_full = np.zeros((h, w), dtype=np.uint8)
    mask_full[ry1:ry2, rx1:rx2] = bp_bin

    col_w = w // N_COLS
    ratios = []
    for i in range(N_COLS):
        col = bp_bin[:, i * col_w:(i + 1) * col_w]
        ratios.append(round((col > 0).sum() / max(col.size, 1), 3))

    danger_y_full = int(h * DANGER_LINE_Y)
    danger_y_local = max(0, danger_y_full - ry1)
    center_x1 = col_w
    center_x2 = 2 * col_w
    if danger_y_local < bp_bin.shape[0]:
        near_strip = bp_bin[danger_y_local:, center_x1:center_x2]
        danger_ratio = round((near_strip > 0).sum() / near_strip.size, 3) if near_strip.size > 0 else 0.0
    else:
        danger_ratio = 1.0

    return {
        'mask': mask_full,
        'ratios': ratios,
        'danger_ratio': danger_ratio,
        'sample_box': (sx1, sy1, sx2, sy2),
    }


# ---------------------------------------------------------------------
#  FSM 2 TRANG THAI
# ---------------------------------------------------------------------
class AvoidanceFSM:
    """
    May trang thai 2 trang thai:
      CRUISE   : di thang, re nhe khi lech
      AVOIDING : khoa mot huong xoay, khong tro lai CRUISE den khi
                 thay duong thoang DU TIN CAY (EXIT_CONFIRM khung lien tiep)
    """
    CRUISE   = "CRUISE"
    AVOIDING = "AVOIDING"

    def __init__(self):
        self.state = self.CRUISE
        self.locked_turn_dir = None    # 'a' hoac 'd' luc vao AVOIDING
        self.danger_streak   = 0       # so khung DANGER lien tiep (de chuyen state)
        self.clear_streak    = 0       # so khung "thoang" lien tiep (de exit AVOIDING)
        self.just_entered    = False   # vua moi vao AVOIDING -> can lui 1 lan
        self.warm_up         = 0       # CAI TIEN: sau khi exit, khong trigger DANGER
                                       # trong WARM_UP_FRAMES khung tiep theo
                                       # (xe co thoi gian di xa khoi vung vua thoat)

    def is_danger(self, info):
        return info['danger_ratio'] < THRESHOLDS['danger']

    def is_clear(self, info):
        """Dieu kien EXIT AVOIDING: giua thoang + danger an toan."""
        left, center, right = info['ratios']
        return (center >= THRESHOLDS['free'] and
                info['danger_ratio'] >= DANGER_SAFE)

    def step(self, info):
        """Tra ve (cmd, reason, state)."""
        ratios = info['ratios']
        left, center, right = ratios

        # === CRUISE ===
        if self.state == self.CRUISE:
            # CAI TIEN: nếu vừa thoát AVOIDING, không trigger DANGER trong vài khung
            # để xe có thời gian đi xa khỏi vùng vừa thoát.
            if self.warm_up > 0:
                self.warm_up -= 1
                self.danger_streak = 0
                if center >= THRESHOLDS['free']:
                    return ('w', "WARM-UP %d/%d, going" % (self.warm_up, WARM_UP_FRAMES), self.state)
                if right > left + STEER_MARGIN:  # uu tien trai -- chi re phai khi phai THOANG HON RO RET
                    return ('a', "WARM-UP %d nudge left" % self.warm_up, self.state)
                return ('d', "WARM-UP %d nudge right" % self.warm_up, self.state)

            if self.is_danger(info):
                self.danger_streak += 1
                if self.danger_streak >= CONFIRM_FRAMES:
                    # Chuyen sang AVOIDING -- chon huong MOT LAN duy nhat
                    self.locked_turn_dir = 'd' if right > left + STEER_MARGIN else 'a'  # uu tien trai
                    self.state = self.AVOIDING
                    self.just_entered = True
                    self.clear_streak = 0
                    self.danger_streak = 0
                    return ('x', "ENTER AVOIDING, lock turn=%s (L=%.2f R=%.2f)" %
                            (self.locked_turn_dir, left, right), self.state)
                # Chua du khung de chuyen -> giu CRUISE, di cham (xoay nhe)
                return ('d' if right > left + STEER_MARGIN else 'a',  # uu tien trai
                        "danger streak %d/%d" % (self.danger_streak, CONFIRM_FRAMES),
                        self.state)
            # Khong danger
            self.danger_streak = 0
            if center >= THRESHOLDS['free']:
                return ('w', "CRUISE: center clear (%.2f)" % center, self.state)
            if left > right + STEER_MARGIN:
                return ('a', "CRUISE: nudge left", self.state)
            if right > left + STEER_MARGIN:
                return ('d', "CRUISE: nudge right", self.state)
            return ('a', "CRUISE: balanced, default left", self.state)

        # === AVOIDING ===
        # just_entered = True nghia la vua chuyen tu CRUISE sang -> lui 1 lan
        if self.just_entered:
            self.just_entered = False
            return ('s', "AVOIDING: entry back-up", self.state)

        # Khong lui them. Chi xoay theo huong da khoa.
        if self.is_clear(info):
            self.clear_streak += 1
            if self.clear_streak >= EXIT_CONFIRM:
                # Du tin cay -> tro ve CRUISE + dat warm_up
                old_dir = self.locked_turn_dir
                self.state = self.CRUISE
                self.locked_turn_dir = None
                self.clear_streak = 0
                self.warm_up = WARM_UP_FRAMES   # CAI TIEN: cooldown sau exit
                return ('w', "EXIT AVOIDING (was=%s) -> CRUISE +warm-up" % old_dir, self.state)
            return ('w', "clear streak %d/%d" %
                    (self.clear_streak, EXIT_CONFIRM), self.state)

        # Chua thoang -> reset clear_streak, tiep tuc xoay huong khoa
        self.clear_streak = 0
        return (self.locked_turn_dir,
                "AVOIDING: turning %s (C=%.2f danger=%.2f)" %
                (self.locked_turn_dir, center, info['danger_ratio']),
                self.state)


# ---------------------------------------------------------------------
#  BUILD VIEW (giu giong cu, them hien thi STATE)
# ---------------------------------------------------------------------
def build_view(frame_bgr, info, cmd, reason, state, fps):
    h, w = frame_bgr.shape[:2]
    original = frame_bgr.copy()
    result = frame_bgr.copy()

    mask = info['mask']
    green_overlay = np.zeros_like(result)
    green_overlay[mask > 0] = (0, 200, 0)
    result = cv2.addWeighted(result, 1.0, green_overlay, 0.45, 0)

    sx1, sy1, sx2, sy2 = info['sample_box']
    cv2.rectangle(result, (sx1, sy1), (sx2, sy2), (0, 255, 255), 1)
    cv2.putText(result, "Floor color", (sx1 + 2, sy1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)

    ratios = info['ratios']
    col_w = w // N_COLS
    for i in range(N_COLS):
        x0 = i * col_w
        cv2.line(result, (x0, 0), (x0, h), (255, 200, 0), 1)
        free = ratios[i] >= THRESHOLDS['free']
        color = (0, 220, 0) if free else (0, 0, 255)
        cv2.putText(result, "%.2f" % ratios[i], (x0 + 6, h - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    dy = int(h * DANGER_LINE_Y)
    danger_triggered = info['danger_ratio'] < THRESHOLDS['danger']
    line_color = (0, 0, 255) if danger_triggered else (0, 200, 255)
    cv2.line(result, (0, dy), (w, dy), line_color, 2)
    cv2.putText(result, "DANGER (%.2f)" % info['danger_ratio'], (w - 130, dy - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, line_color, 1)

    _label(original, "VIDEO")
    _label(result, "RESULT")

    info_h = 110
    canvas = np.zeros((h + info_h, w * 2, 3), dtype=np.uint8)
    canvas[0:h, 0:w] = original
    canvas[0:h, w:w * 2] = result

    state_color = (0, 220, 0) if state == "CRUISE" else (0, 100, 255)
    cmd_color = {'w': (0, 220, 0), 'x': (0, 0, 255),
                 'a': (0, 200, 255), 'd': (0, 200, 255),
                 's': (200, 200, 0)}.get(cmd, (255, 255, 255))

    cv2.putText(canvas, "STATE: %s" % state, (12, h + 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, state_color, 2)
    cv2.putText(canvas, "CMD: %s" % cmd, (260, h + 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, cmd_color, 2)
    cv2.putText(canvas, reason, (12, h + 56),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)
    cv2.putText(canvas, "FPS: %.1f" % fps, (w * 2 - 120, h + 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
    cv2.putText(canvas, "FSM 2-state | ACT=%.2fs CONFIRM=%d EXIT=%d" %
                (ACT_DURATION, CONFIRM_FRAMES, EXIT_CONFIRM),
                (12, h + 86),
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

    if AUTO_CALIBRATE and '--manual' not in sys.argv:
        calibrate_thresholds(cap)
    else:
        print("[run] MANUAL: floor=%d free=%.2f danger=%.2f" %
              (THRESHOLDS['floor'], THRESHOLDS['free'], THRESHOLDS['danger']))

    motor = MotorController(USE_ARDUINO, ARDUINO_PORT, ARDUINO_BAUD)
    fsm   = AvoidanceFSM()

    writer = None
    if SAVE_VIDEO:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, 10.0, (FRAME_W * 2, FRAME_H + 78))

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

            # [MOI] KHOAN NGHI: doc CONFIRM_PAUSE_FRAMES truoc khi giao cho FSM.
            # Chi dung detect_floor() (sensor thuan tuy), KHONG goi fsm.step() trong pause.
            # Lay trung binh cac chi so sensor -> giam nhieu -> giao FSM quyet dinh 1 lan duy nhat.
            sensor_readings = []
            last_frame = frame
            for _ in range(CONFIRM_PAUSE_FRAMES):
                ok2, frame2 = cap.read()
                if ok2 and frame2 is not None:
                    sensor_readings.append(detect_floor(frame2))
                    last_frame = frame2
            if not sensor_readings:
                sensor_readings = [detect_floor(frame)]

            # Trung binh hoa cac chi so sensor (ratios, danger_ratio)
            avg_ratios = [
                sum(r['ratios'][i] for r in sensor_readings) / len(sensor_readings)
                for i in range(N_COLS)
            ]
            avg_danger = sum(r['danger_ratio'] for r in sensor_readings) / len(sensor_readings)
            info = {
                'ratios':      avg_ratios,
                'danger_ratio': round(avg_danger, 3),
                'mask':        sensor_readings[-1]['mask'],       # dung mask frame cuoi de hien thi
                'sample_box':  sensor_readings[-1]['sample_box'],
            }
            frame = last_frame

            # Giao FSM quyet dinh 1 lan duy nhat voi sensor da lam muot
            cmd, reason, state = fsm.step(info)
            proc_time = time.time() - t0
            fps = 1.0 / max(proc_time, 1e-6)

            view = build_view(frame, info, cmd, reason, state, fps)
            if STREAM_ENABLE: _frame_buffer.update(view)
            if writer is not None: writer.write(view)

            # Gui lenh + sleep theo loai lenh
            motor.send(cmd)
            if cmd == 's':
                time.sleep(ENTRY_BACK_S)         # lui lau hon
            elif state == "AVOIDING" and cmd in ('a', 'd'):
                time.sleep(TURN_NUDGE_S)         # xoay nhip nho trong AVOIDING
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
