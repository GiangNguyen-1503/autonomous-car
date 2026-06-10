import time
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

#--------------
CAM_INDEX   = 0 #front cam
FRAME_W     = 320
FRAME_H     = 240
DISPLAY_SCALE = 2

#auto-calibration
AUTO_CALIBRATE = True
CALIB_SECONDS = 3.0

#floor
FLOOR_THRESH    = 40
FREE_MIN    = 0.45
STEER_MARGIN    = 0.08
N_COLS      = 3

DANGER_LINE_Y = 0.70
DANGER_MIN = 0.30

ACT_DURATION    = 0.3 #strop and go timing
SETTLE_FRAMES   = 3

#backward
ESCAPE_BACK_S = 0.4
ESCAPE_TURN_S = 0.6

#hardware
USE_ARDUINO = True
ARDUINO_PORT    = '/dev/ttyACM0'
ARDUINO_BAUD    = 9600

#save file and stream
SAVE_VIDEO  = True
OUTPUT_PATH = 'run_output.avi'
STREAM_ENABLE   = True
STREAM_PORT = 8000
#--------------

THRESHOLDS = {
    'floor': FLOOR_THRESH,
    'free': FREE_MIN,
    'danger': DANGER_MIN,
}

def calibrate_thresholds(cap, seconds=CALIB_SECONDS):
    print("[calibrate] START CALIBRATING")
    for i in range(3, 0, -1):
        print("[calibrate] START...")
        time.sleep(1)
    print("[calibrate] calibrate environment")

    bp_samples = []
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
    sx1, sx2 = int(w * 0.30), int(w * 0.70)
    sy1, sy2 = int(h * 0.85), h
    floor = hsv[sy1:sy2, sx1:sx2]
    hist = cv2.calcHist([floor], [0, 1], None, [30, 32], [0, 180, 0, 256])
    cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
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

    #results
    if not otsu_samples:
        print("[calibrate] ERROR: CAN NOT READ FRAME.. Stop fallback.")
        return

    new_floor  = int(round(float(np.median(otsu_samples))))

    #ratios on 3 cols
    avg_ratio  = float(np.mean([np.mean(r) for r in ratios_samples]))
    new_free   = round(avg_ratio * 0.7, 2)
    new_danger = round(float(np.mean(danger_samples)) * 0.5, 2) if danger_samples else THRESHOLDS['danger']

    #protect
    new_floor  = max(15, min(120, new_floor))
    new_free   = max(0.20, min(0.80, new_free))
    new_danger = max(0.10, min(0.60, new_danger))

    THRESHOLDS['floor']  = new_floor
    THRESHOLDS['free']   = new_free
    THRESHOLDS['danger'] = new_danger

    print("[calibrate] === CALIBRATION RESULTST ===")
    print("[calibrate] FLOOR_THRESH = %d (default: %d)" % (new_floor, FLOOR_THRESH))
    print("[calibrate] FREE_MIN     = %.2f (default: %.2f)" % (new_free, FREE_MIN))
    print("[calibrate] DANGER_MIN   = %.2f (default: %.2f)" % (new_danger, DANGER_MIN))



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
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    return server


#arduino with wasd control
class MotorController:
    def __init__(self, use_arduino, port, baud):
        self.ser = None
        self.last_cmd = None
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

    #histogram floor
    sx1, sx2 = int(w * 0.30), int(w * 0.70)
    sy1, sy2 = int(h * 0.85), h
    floor = hsv[sy1:sy2, sx1:sx2]
    hist = cv2.calcHist([floor], [0, 1], None, [30, 32], [0, 180, 0, 256])
    cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)

    #back-projection
    rx1, rx2 = 0, w
    ry1, ry2 = int(h * 0.50), sy1
    region = hsv[ry1:ry2, rx1:rx2]
    bp = cv2.calcBackProject([region], [0, 1], hist, [ 0, 180, 0, 256], scale=1)
    bp = cv2.medianBlur(bp, 5)

    #binary mask
    _, bp_bin = cv2.threshold(bp, FLOOR_THRESH, 255, cv2.THRESH_BINARY)

    #mask scale
    mask_full = np.zeros((h, w), dtype=np.uint8)
    mask_full[ry1:ry2, rx1:rx2] = bp_bin

    #floor ratios each col
    col_w = w // N_COLS
    ratios = []
    for i in range(N_COLS):
        col = bp[:, i * col_w:(i + 1) * col_w]
        ratios.append(round((col > FLOOR_THRESH).sum() / max(col.size, 1), 3))

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
    return {
        'mask': mask_full,
        'ratios': ratios,
        'danger_ratio': danger_ratio,
        'sample_box': (sx1, sy1, sx2, sy2),
        'scan_box':   (rx1, ry1, rx2, ry2),
    }


#think
def decide(frame_bgr):
    info = detect_floor(frame_bgr)
    ratios = info['ratios']

    #check danger
    if info['danger_ratio'] < THRESHOLDS['danger']:
        cmd = 'x'
        reason = "DANGER: TOO CLOSE = %.2f < %.2f" % (info['danger_ratio'], THRESHOLDS['danger'])
    else:
        cmd = _steer_from_ratios(ratios)
        reason = "FLOOR L/C/R = %s | close = %.2f" % (ratios, info['danger_ratio'])
    info['cmd'] = cmd
    info['reason'] = reason
    return info


def _steer_from_ratios(ratios):
    left, center, right = ratios
    if center >= THRESHOLDS['free']:
        return 'w'	#wasd/flbr
    #if max(left, right) < FREE_MIN:
        #return 'x'	#Stop/x
    if left > right + STEER_MARGIN:
        return 'a'
    if right > left + STEER_MARGIN:
        return 'd'
    return 'a'

#show up
def build_view(frame_bgr, decision, fps):
    h, w = frame_bgr.shape[:2]
    original = frame_bgr.copy()
    result = frame_bgr.copy()

    #paint green - floor
    mask = decision['mask']
    green_overlay = np.zeros_like(result)
    green_overlay[mask > 0] = (0, 200, 0)
    result = cv2.addWeighted(result, 1.0, green_overlay, 0.45, 0)

    #floor area
    sx1, sy1, sx2, sy2 = decision['sample_box']
    cv2.rectangle(result, (sx1, sy1), (sx2, sy2), (0, 255, 255), 1)
    cv2.putText(result, "Floor color", (sx1 + 2, sy1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)

    #draw lines
    ratios = decision['ratios']
    col_w = w // N_COLS
    for i in range(N_COLS):
        x0 = i * col_w
        cv2.line(result, (x0, 0), (x0, h), (255, 200, 0), 1)
        free  = ratios[i] >= FREE_MIN
        color = (0, 220, 0) if free else (0, 0, 255)
        cv2.putText(result, "%.2f" % ratios[i], (x0 + 6, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    #danger line
    dy = int(h * DANGER_LINE_Y)
    danger_triggered = decision['danger_ratio'] < THRESHOLDS['danger']
    line_color = (0, 0, 255) if danger_triggered else (0, 200, 255)
    cv2.line(result, (0, dy), (w, dy), line_color, 2)
    cv2.putText(result, "DANGER (%.2f)" % decision['danger_ratio'], (w - 130, dy - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, line_color, 1)


    #label
    _label(original, "VIDEO")
    _label(result,   "RESULT")

    #info
    info_h = 90
    canvas = np.zeros((h + info_h, w * 2, 3), dtype=np.uint8)
    canvas[0:h, 0:w] = original
    canvas[0:h, w:w * 2] = result

    cmd = decision['cmd']
    cmd_color = {'w - front': (0, 220, 0), 'x - stop': (0, 0, 255), 'a - left': (0, 200, 255), 'd - right': (0, 200, 255)}.get(cmd, (255, 255, 255))
    cv2.putText(canvas, "CMD: %s" % cmd, (12, h + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.85, cmd_color, 2)
    cv2.putText(canvas, decision['reason'], (12, h + 62), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
    cv2.putText(canvas, "FPS proc: %.1f" % fps, (w * 2 - 160, h + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
    cv2.putText(canvas, "stop-and-go | ACT=%.2fs" % ACT_DURATION, (w * 2 - 220, h + 62), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (140, 140, 140), 1)

    #
    if DISPLAY_SCALE != 1:
        canvas = cv2.resize(canvas, (canvas.shape[1] * DISPLAY_SCALE, canvas.shape[0] * DISPLAY_SCALE), interpolation=cv2.INTER_LINEAR,)

    return canvas

def _label(img, text):
    cv2.rectangle(img, (0, 0), (img.shape[1], 22), (40, 40, 40), -1)
    cv2.putText(img, text, (8, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

#main loop
def main():
    use_auto_calibrate = AUTO_CALIBRATE
    if '--manual' in sys.argv:
        use_auto_calibrate = False
        #print("[run] --manual flag")

    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    if not cap.isOpened():
        print("[ERROR] CANNOT OPEN camera index", CAM_INDEX)
        sys.exit(1)

    #environment calibration
    if use_auto_calibrate:
        calibrate_thresholds(cap)
    else:
        print("[run] STOP: floor=%d free=%.2f danger=%.2f" % (THRESHOLDS['floor'], THRESHOLDS['free'], THRESHOLDS['dangger']))

    motor = MotorController(USE_ARDUINO, ARDUINO_PORT, ARDUINO_BAUD)

    writer = None
    if SAVE_VIDEO:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out_w  = FRAME_W * 2 * DISPLAY_SCALE
        out_h  = (FRAME_H + 90) * DISPLAY_SCALE
        writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, 10.0, (FRAME_W * 2, FRAME_H + 78))

    if STREAM_ENABLE:
        start_stream_server(STREAM_PORT)
        print("[stream]   http://<ip-address>:%d" % STREAM_PORT)
        print("[stream]ip: hostname -I)")

    print("[run]")

    escape_turn_dir = 'a'

    try:
        while True:
            #sense
            ok, frame = cap.read()
            if not ok or frame is None:
                print("[ERROR] CANNOT READ FRAME")
                break

            #think
            t0 = time.time()
            decision = decide(frame)
            proc_time = time.time() - t0
            fps = 1.0 / max(proc_time, 1e-6)
            cmd = decision['cmd']

            #update
            view = build_view(frame, decision, fps)
            if STREAM_ENABLE:
                _frame_buffer.update(view)
            if writer is not None:
                writer.write(view)

            #act
            if cmd == 'x':
                print("[escape] DANGER -- lui %.1fs roi quay %s %.1fs" % (ESCAPE_BACK_S, escape_turn_dir, ESCAPE_TURN_S))
                motor.send('s')
                time.sleep(ESCAPE_BACK_S)
                motor.send(escape_turn_dir)
                time.sleep(ESCAPE_TURN_S)
                escape_turn_dir = 'd' if escape_turn_dir == 'a' else 'a'
                continue

            motor.send(cmd)
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
