import time
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

#--------------
CAM_INDEX = 0 #front cam
FRAME_W = 320
FRAME_H = 240
DISPLAY_SCALE = 2

#auto-calibration
AUTO_CALIBRATE = True
CALIB_SECONDS = 3.0

#floor
FLOOR_THRESH = 40
FREE_MIN = 0.45
STEER_MARGIN = 0.15
N_COLS = 3

DANGER_LINE_Y = 0.70
DANGER_MIN = 0.3
DANGER_SAFE = 0.45

#fsm timing, confirm
ACT_DURATION = 0.3 #strop and go timing
CONFIRM_FRAMES = 4
CONFIRM_PAUSE_FRAMES = 3 
EXIT_CONFIRM = 3

#SETTLE_FRAMES = 3
ENTRY_BACK_S = 0.4
TURN_NUDGE_S = 0.3
WARM_UP_FRAMES = 3

#backward
ESCAPE_BACK_S = 0.4
ESCAPE_TURN_S = 0.6

#hardware
USE_ARDUINO = True
ARDUINO_PORT = '/dev/ttyACM0'
ARDUINO_BAUD = 9600

#save file and stream
SAVE_VIDEO = True
OUTPUT_PATH = 'run_output.avi'
STREAM_ENABLE = True
STREAM_PORT = 8000
#--------------

THRESHOLDS = {
    'floor': FLOOR_THRESH,
    'free': FREE_MIN,
    'danger': DANGER_MIN,
    'floor_hist': None,
    'poison_threshold': 0.0,
}

CALIB_MIN_RATIO = 0.20

POISON_STD_K = 3.0

#auto calibrate
def calibrate_thresholds(cap, seconds=CALIB_SECONDS):
    print("[calibrate] START CALIBRATING")
    for i in range(3, 0, -1):
        print("[calibrate] START...")
        time.sleep(1)
    print("[calibrate] measuring...")

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

        #back-projection
        h, w = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

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

        #otsu
        otsu_thresh, bp_bin = cv2.threshold(bp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        otsu_samples.append(otsu_thresh)

        #calculate ratios
        col_w = w // N_COLS
        r = []
        for i in range(N_COLS):
            col = bp_bin[:, i * col_w:(i + 1) * col_w]
            r.append((col > 0).sum() / max(col.size, 1))
        ratios_samples.append(r)

        # calculate danger_ratio
        danger_y = max(0, int(h * DANGER_LINE_Y) - ry1)
        if danger_y < bp_bin.shape[0]:
            near = bp_bin[danger_y:, col_w:2 * col_w]
            if near.size > 0:
                danger_samples.append((near > 0).sum() / near.size)

        #bp_mean
        sample_bp = cv2.calcBackProject([floor], [0, 1], hist, [0, 180, 0, 256], scale=1)
        sample_bp_means.append(float(np.mean(sample_bp)))


    #results
    if not otsu_samples:
        print("[calibrate] ERROR: CAN NOT READ FRAME.. Stop fallback.")
        return False

    #ratios on 3 cols
    avg_ratio = float(np.mean([np.mean(r) for r in ratios_samples]))

    #sanity check
    if avg_ratio < CALIB_MIN_RATIO:
        print("[calibrate] REJECT: avg_ratio = %.2f < %.2f" % (avg_ratio, CALIB_MIN_RATIO))
        print("[calibrate] -> NOT FLOOR COLOR.")
        return False

    #threshold update
    new_floor  = max(15, min(120, int(round(float(np.median(otsu_samples))))))
    new_free = max(0.20, min(0.80, round(avg_ratio * 0.7, 2)))
    new_danger = round(float(np.mean(danger_samples)) * 0.5, 2) if danger_samples else DANGER_MIN

    #protect
    new_danger = max(0.10, min(0.60, new_danger))

    THRESHOLDS['floor']  = new_floor
    THRESHOLDS['free']   = new_free
    THRESHOLDS['danger'] = new_danger

    #2.avg histogram save
    avg_hist = np.mean(np.stack(hist_samples), axis=0)
    cv2.normalize(avg_hist, avg_hist, 0, 255, cv2.NORM_MINMAX)
    THRESHOLDS['floor_hist'] = avg_hist

    #3.poisoned detection
    bp_mean = float(np.mean(sample_bp_means))
    bp_std = float(np.std(sample_bp_means))
    THRESHOLDS['poison_threshold'] = max(0.0, bp_mean - POISON_STD_K * bp_std)
    #THRESHOLDS['poison_threshold'] = poison_threshold

    print("[calibrate] DONE: floor=%d free=%.2f danger=%.2f" % (new_floor, new_free, new_danger))
    print("[calibrate] bp_mean=%.1f std=%.1f poison_thresh=%.1f" % (bp_mean, bp_std, THRESHOLDS['poison_threshold']))

    return True

#stream mjeg
class FrameBuffer:
    def __init__(self):
        self._jpeg = None
        self._lock = threading.Lock()

    def update(self, bgr_image):
        ok, buf = cv2.imencode('.jpg', bgr_image, [cv2.IMWRITE_JPEG_QUALITY, 75])
        if ok:
            with self._lock:
                self._jpeg = buf.tobytes()

    def get(self):
        with self._lock:
            return self._jpeg

_frame_buffer = FrameBuffer()

class _MJPEGHandler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
        self.end_headers()
        try:
            while True:
                jpeg = _frame_buffer.get()
                if jpeg is None:
                    time.sleep(0.05)
                    continue
                self.wfile.write(b'--frame\r\n')
                self.send_header('Content-Type', 'image/jpeg')
                self.send_header('Content-Length', str(len(jpeg)))
                self.end_headers()
                self.wfile.write(jpeg)
                self.wfile.write(b'\r\n')
                time.sleep(0.1)
        except (BrokenPipeError, ConnectionResetError):
            pass

def start_stream_server(port):
    server = ThreadingHTTPServer(('0.0.0.0', port), _MJPEGHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server

#arduino with wasd control
class MotorController:
    def __init__(self, use_arduino, port, baud):
        self.ser = None
        self.last_cmd = None
        self.prev_cmd = None
        if use_arduino:
            try:
                import serial
                self.ser = serial.Serial(port, baud, timeout=1)
                time.sleep(2)
                print("[motor] Arduino CONNECTED AT:", port)
            except Exception as e:
                print("MOTOR cannot connect:", e)
        else:
            print("MOTOR USE_ARDUINO = FALSE")

    def send(self, cmd):
        if cmd == self.last_cmd:
            return
        self.prev_cmd = self.last_cmd

        print("MOTOR ->", cmd)
        self.last_cmd = cmd
        if self.ser is not None:
            try:
                self.ser.write(cmd.encode())
            except Exception as e:
                print("MOTOR SERIAL ERROR:", e)

    def stop(self):
        self.send('x')

    def close(self):
        if self.ser is not None:
            self.stop()
            self.ser.close()


#detection
def detect_floor(frame_bgr):
    h, w = frame_bgr.shape[:2]
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    hsv = cv2.bilateralFilter(hsv, 5, 50, 50)

    #histogram floor
    sx1, sx2 = int(w * 0.30), int(w * 0.70)
    sy1, sy2 = int(h * 0.85), h
    floor = hsv[sy1:sy2, sx1:sx2]

    #2.use calibrated histogram
    if THRESHOLDS['floor_hist'] is not None:
        hist = THRESHOLDS['floor_hist']
    else:
        hist = cv2.calcHist([floor], [0, 1], None, [30, 32], [0, 180, 0, 256])
        cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)

    #back-projection
    rx1, rx2 = 0, w
    ry1, ry2 = int(h * 0.50), sy1
    region = hsv[ry1:ry2, rx1:rx2]
    bp = cv2.calcBackProject([region], [0, 1], hist, [ 0, 180, 0, 256], scale=1)
    bp = cv2.medianBlur(bp, 5)
    #binary mask
    _, bp_bin = cv2.threshold(bp, THRESHOLDS['floor'], 255, cv2.THRESH_BINARY)

    #mask scale
    mask_full = np.zeros((h, w), dtype=np.uint8)
    mask_full[ry1:ry2, rx1:rx2] = bp_bin

    #floor ratios each col
    col_w = w // N_COLS
    ratios = []
    for i in range(N_COLS):
        col = bp_bin[:, i * col_w:(i + 1) * col_w]
        ratios.append(round((col > 0).sum() / max(col.size, 1), 3))

    #danger
    danger_y_full = int(h* DANGER_LINE_Y)
    danger_y_local = max(0, danger_y_full - ry1)
    center_x1 = col_w
    center_x2 = 2 * col_w
    if danger_y_local < bp_bin.shape[0]:
        near_strip = bp_bin[danger_y_local:, center_x1:center_x2]
        if near_strip.size > 0:
            danger_ratio = round((near_strip > 0).sum() / near_strip.size, 3)
        else:
            danger_ratio = 0.0
    else:
        danger_ratio = 1.0

    #3.
    #sample_bp = cv2.calcBackProject([floor], [0, 1], hist, [0, 180, 0, 256], scale=1)
    #sample_bp_mean = float(np.mean(sample_bp))
    #poison_thresh = THRESHOLDS['poison_threshold']
    #poisoned = False
    #if poison_thresh > 0:
    #    poisoned = sample_bp_mean < poison_thresh

    return {
        'mask': mask_full,
        'ratios': ratios,
        'danger_ratio': danger_ratio,
        'sample_box': (sx1, sy1, sx2, sy2),
    }

#fsm 2 state
class AvoidanceFSM:
    CRUISE   = "CRUISE"
    AVOIDING = "AVOIDING"

    def __init__(self):
        self.state = self.CRUISE
        self.locked_turn_dir = None
        self.danger_streak = 0
        self.clear_streak = 0
        self.just_entered = False
        self.warm_up = 0

    def is_danger(self, info):
        return info['danger_ratio'] < THRESHOLDS['danger']

    def is_clear(self, info):
        left, center, right = info['ratios']
        return (center >= THRESHOLDS['free'] and info['danger_ratio'] >= DANGER_SAFE)

    def step(self, info):
        ratios = info['ratios']
        left, center, right = ratios

        #cruise
        if self.state == self.CRUISE:
            if self.warm_up > 0:
                self.warm_up -= 1
                self.danger_streak = 0
                if center >= THRESHOLDS['free']:
                    return ('w', "WARM-UP %d/%d, going" % (self.warm_up, WARM_UP_FRAMES), self.state)
                if right > left + STEER_MARGIN:
                    return ('a', "WARM-UP %d nudge left" % self.warm_up, self.state)
                return ('d', "WARM-UP %d nudge right" % self.warm_up, self.state)

            if self.is_danger(info):
                self.danger_streak += 1
                if self.danger_streak >= CONFIRM_FRAMES:
                    self.locked_turn_dir = 'd' if right > left + STEER_MARGIN else 'a'  # uu tien trai
                    self.state = self.AVOIDING
                    self.just_entered = True
                    self.clear_streak = 0
                    self.danger_streak = 0
                    return ('x', "ENTER AVOIDING, lock turn=%s (L=%.2f R=%.2f)" % (self.locked_turn_dir, left, right), self.state)
                return ('d' if right > left + STEER_MARGIN else 'a', "danger streak %d/%d" % (self.danger_streak, CONFIRM_FRAMES), self.state)

            #no danger
            self.danger_streak = 0
            if center >= THRESHOLDS['free']:
                return ('w', "CRUISE: center clear (%.2f)" % center, self.state)
            if left > right + STEER_MARGIN:
                return ('a', "CRUISE: nudge left", self.state)
            if right > left + STEER_MARGIN:
                return ('d', "CRUISE: nudge right", self.state)
            return ('a', "CRUISE: balanced, default left", self.state)

        #avoiding
        if self.just_entered:
            self.just_entered = False
            return ('s', "AVOIDING: entry back-up", self.state)

        if self.is_clear(info):
            self.clear_streak += 1
            if self.clear_streak >= EXIT_CONFIRM:
                old_dir = self.locked_turn_dir
                self.state = self.CRUISE
                self.locked_turn_dir = None
                self.clear_streak = 0
                self.warm_up = WARM_UP_FRAMES
                return ('w', "EXIT AVOIDING (was=%s) -> CRUISE +warm-up" % old_dir, self.state)
            return ('w', "clear streak %d/%d" % (self.clear_streak, EXIT_CONFIRM), self.state)
        #not clear yet
        self.clear_streak = 0
        return (self.locked_turn_dir, "AVOIDING: turning %s (C=%.2f danger=%.2f)" % (self.locked_turn_dir, center, info['danger_ratio']), self.state)


#show up
def build_view(frame_bgr, info, cmd, reason, state, fps):
    h, w = frame_bgr.shape[:2]
    original = frame_bgr.copy()
    result = frame_bgr.copy()

    #paint green - floor
    mask = info['mask']
    green_overlay = np.zeros_like(result)
    green_overlay[mask > 0] = (0, 200, 0)
    result = cv2.addWeighted(result, 1.0, green_overlay, 0.45, 0)

    #floor area
    sx1, sy1, sx2, sy2 = info['sample_box']
    cv2.rectangle(result, (sx1, sy1), (sx2, sy2), (0, 255, 255), 1)
    cv2.putText(result, "Floor color", (sx1 + 2, sy1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)

    #draw lines
    ratios = info['ratios']
    col_w = w // N_COLS
    for i in range(N_COLS):
        x0 = i * col_w
        cv2.line(result, (x0, 0), (x0, h), (255, 200, 0), 1)
        free = ratios[i] >= THRESHOLDS['free']
        color = (0, 220, 0) if free else (0, 0, 255)
        cv2.putText(result, "%.2f" % ratios[i], (x0 + 6, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    #danger line
    dy = int(h * DANGER_LINE_Y)
    danger_triggered = info['danger_ratio'] < THRESHOLDS['danger']
    line_color = (0, 0, 255) if danger_triggered else (0, 200, 255)
    cv2.line(result, (0, dy), (w, dy), line_color, 2)
    cv2.putText(result, "DANGER (%.2f)" % info['danger_ratio'], (w - 130, dy - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, line_color, 1)


    #label
    _label(original, "VIDEO")
    _label(result, "RESULT")

    #info
    info_h = 110
    canvas = np.zeros((h + info_h, w * 2, 3), dtype=np.uint8)
    canvas[0:h, 0:w] = original
    canvas[0:h, w:w * 2] = result

    state_color = (0, 220, 0) if state == "CRUISE" else (0, 100, 255)

    cmd_color = {'w': (0, 220, 0), 'x': (0, 0, 255), 'a': (0, 200, 255), 'd': (0, 200, 255), 's': (200, 200, 0)}.get(cmd, (255, 255, 255))

    cv2.putText(canvas, "STATE: %s" % state, (12, h + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, state_color, 2)
    cv2.putText(canvas, "CMD: %s" % cmd, (260, h + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, cmd_color, 2)
    cv2.putText(canvas, reason, (12, h + 56), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)
    cv2.putText(canvas, "FPS: %.1f" % fps, (w * 2 - 120, h + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
    cv2.putText(canvas, "FSM 2-state | ACT=%.2fs CONFIRM=%d EXIT=%d" % (ACT_DURATION, CONFIRM_FRAMES, EXIT_CONFIRM), (12, h + 86), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (140, 140, 140), 1)

    #
    if DISPLAY_SCALE != 1:
        canvas = cv2.resize(canvas, (canvas.shape[1] * DISPLAY_SCALE, canvas.shape[0] * DISPLAY_SCALE), interpolation=cv2.INTER_LINEAR,)

    return canvas

def _label(img, text):
    cv2.rectangle(img, (0, 0), (img.shape[1], 22), (40, 40, 40), -1)
    cv2.putText(img, text, (8, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

#main loop
def main():
    #use_auto_calibrate = AUTO_CALIBRATE
    #if '--manual' in sys.argv:
    #    use_auto_calibrate = False
    #    print("[run] --manual flag")

    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    if not cap.isOpened():
        print("[ERROR] CANNOT OPEN camera index", CAM_INDEX)
        sys.exit(1)

    #environment calibration
    if AUTO_CALIBRATE and '--manual' not in sys.argv:
        calibrate_thresholds(cap)
    else:
        print("[run] STOP: floor=%d free=%.2f danger=%.2f" % (THRESHOLDS['floor'], THRESHOLDS['free'], THRESHOLDS['danger']))

    motor = MotorController(USE_ARDUINO, ARDUINO_PORT, ARDUINO_BAUD)
    fsm = AvoidanceFSM()

    writer = None
    if SAVE_VIDEO:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        #out_w  = FRAME_W * 2 * DISPLAY_SCALE
        #out_h  = (FRAME_H + 90) * DISPLAY_SCALE
        writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, 10.0, (FRAME_W * 2, FRAME_H + 78))

    if STREAM_ENABLE:
        start_stream_server(STREAM_PORT)
        print("[stream]   http://<ip-address>:%d" % STREAM_PORT)
        #print("[stream]ip: hostname -I)")

    print("[run] START")

    #escape_turn_dir = 'a'

    try:
        while True:
            #sense
            ok, frame = cap.read()
            if not ok or frame is None:
                print("[ERROR] CANNOT READ FRAME")
                break

            #think
            t0 = time.time()
            info = detect_floor(frame)
            cmd, reason, state = fsm.step(info)
            proc_time = time.time() - t0
            fps = 1.0 / max(proc_time, 1e-6)

            if motor.last_cmd is not None and cmd != motor.last_cmd:
                motor.send('x')
                print("[pause] cmd change: %s -> %s, confirming.." % (motor.last_cmd, cmd))
                confirmed_cmd = cmd
                for _ in range(CONFIRM_PAUSE_FRAMES):
                    ok2, frame2 = cap.read()
                    if ok2 and frame2 is not None:
                        info2 = detect_floor(frame2)
                        confirmed_cmd, reason, state = fsm.step(info2)
                        frame = frame2
                        info = info2
                cmd = confirmed_cmd
                print("[pause] confirmed cmd: %s" % cmd)
                fps = 1.0 / max(time.time() - t0, 1e-6) 

            #update
            view = build_view(frame, info, cmd, reason, state, fps)
            if STREAM_ENABLE:
                _frame_buffer.update(view)
            if writer is not None:
                writer.write(view)

            #act
            motor.send(cmd)
            if cmd == 's':
                time.sleep(ENTRY_BACK_S)
            elif state == "AVOIDING" and cmd in ('a', 'd'):
                time.sleep(TURN_NUDGE_S)
            else:
                time.sleep(ACT_DURATION)

    except KeyboardInterrupt:
        print("\n[RUN] Stop")
    finally:
        motor.close()
        cap.release()
        if writer is not None:
            writer.release()
            print("[RUN Video saved:", OUTPUT_PATH)
        print("[RUN] FINISH")

if __name__ == "__main__":
    main()
