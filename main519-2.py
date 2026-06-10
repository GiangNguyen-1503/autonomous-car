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
FLOOR_THRESH = 20
FREE_MIN = 0.45
STEER_MARGIN = 0.08
N_COLS = 3

DANGER_LINE_Y = 0.70
DANGER_MIN = 0.30

ACT_DURATION = 0.3 #strop and go timing
SETTLE_FRAMES = 3

#backward
ESCAPE_BACK_S = 0.4
ESCAPE_TURN_S = 0.6

#hardware
USE_ARDUINO = False
ARDUINO_PORT = '/dev/ttyACM0'
ARDUINO_BAUD = 9600

#save file and stream
SAVE_VIDEO = True
OUTPUT_PATH = 'run_output.avi'
STREAM_ENABLE = True
STREAM_PORT = 8000
#--------------

#new obstacle detection params
#edge detection
EDGE_CANNY_LOW = 50
EDGE_CANNY_HIGH = 100
EDGE_BLUR_KERNEL = 5
EDGE_MIN_LENGTH = 15
EDGE_SENSITIVITY = 0.35

#contour detection
CONTOUR_MIN_AREA = 100
CONTOUR_APPROX_EPS = 0.08
CONTOUR_SENSITIVITY = 0.25

#multilayer fusion weights
WEIGHT_FLOOR = 0.50
WEIGHT_EDGE = 0.30
WEIGHT_CONTOUR = 0.20

THRESHOLDS = {
    'floor': FLOOR_THRESH,
    'free': FREE_MIN,
    'danger': DANGER_MIN,
    'floor_hist': None,
    'poison_threshold': 0.0,
    'edge_threshold': 0.20,
    'contour_threshold': 0.15,
}

CALIB_MIN_RATIO = 0.50
POISON_STD_K = 2.0

def calibrate_thresholds(cap, seconds=CALIB_SECONDS):
    print("[calibrate] START CALIBRATING")
    for i in range(3, 0, -1):
        print("[calibrate] START...")
        time.sleep(1)
    print("[calibrate] calibrate environment")

    hist_samples = []
    sample_bp_means = []
    ratios_samples = []
    danger_samples = []
    otsu_samples = []
    edge_samples = []

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

        #bp_mean
        sample_bp = cv2.calcBackProject([floor], [0, 1], hist, [0, 180, 0, 256], scale=1)
        sample_bp_means.append(float(np.mean(sample_bp)))

        #edge detetion calibration
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 9, 30, 30)
        edges = cv2.Canny(gray, EDGE_CANNY_LOW, EDGE_CANNY_HIGH)
        edge_ratio = (edges > 0).sum() / max(edges.size, 1)
        edge_samples.append(edge_ratio)

    #results
    if not otsu_samples:
        print("[calibrate] ERROR: CAN NOT READ FRAME.. Stop fallback.")
        return

    #ratios on 3 cols
    avg_ratio = float(np.mean([np.mean(r) for r in ratios_samples]))

    #sanity check
    if avg_ratio < CALIB_MIN_RATIO:
        print("[calibrate] REJECT: avg_ratio = %.2f < %.2f" % (avg_ratio, CALIB_MIN_RATIO))
        print("[calibrate] -> NOT FLOOR COLOR.")
        return False

    #edge threshold calibration
    new_edge = round(float(np.mean(edge_samples)) * 0.6, 3)

    #threshold update
    new_floor  = int(round(float(np.median(otsu_samples))))
    new_free = round(avg_ratio * 0.7, 2)
    new_danger = round(float(np.mean(danger_samples)) * 0.5, 2) if danger_samples else THRESHOLDS['danger']

    #protect
    new_floor = max(15, min(120, new_floor))
    new_free = max(0.20, min(0.80, new_free))
    new_danger = max(0.10, min(0.60, new_danger))
    new_edge = max(0.05, min(0.50, new_edge))

    THRESHOLDS['floor'] = new_floor
    THRESHOLDS['free'] = new_free
    THRESHOLDS['danger'] = new_danger
    THRESHOLDS['edge_threshold'] = new_edge

    #2.avg histogram save
    avg_hist = np.mean(np.stack([cv2.calcHist([hsv[sy1:sy2, sx1:sx2]], [0, 1], None, [30, 32], [0, 180, 0, 256]) for _ in range(len(hist_samples)) if len(hist_samples) > 0]), axis=0) if hist_samples else np.zeros((30, 32))
    if avg_hist.size > 0:
        cv2.normalize(avg_hist, avg_hist, 0, 255, cv2.NORM_MINMAX)
    THRESHOLDS['floor_hist'] = avg_hist

    #3.poisoned detection
    bp_mean = float(np.mean(sample_bp_means))
    bp_std = float(np.std(sample_bp_means))
    poison_threshold = max(0.0, bp_mean - POISON_STD_K * bp_std)
    THRESHOLDS['poison_threshold'] = poison_threshold

    print("[calibrate] === CALIBRATION RESULTST ===")
    print("[calibrate] FLOOR_THRESH = %d (default: %d)" % (new_floor, FLOOR_THRESH))
    print("[calibrate] FREE_MIN     = %.2f (default: %.2f)" % (new_free, FREE_MIN))
    print("[calibrate] DANGER_MIN   = %.2f (default: %.2f)" % (new_danger, DANGER_MIN))
    print("[calibrate] EDGE_THRESH  = %.3f (NEW)" % new_edge)
    print("[calibrate] avg_ratio    = %.2f  (sanity OK >= %.2f)" % (avg_ratio, CALIB_MIN_RATIO))
    print("[calibrate] bp_mean (floor) = %.1f  (std=%.1f)" % (bp_mean, bp_std))
    print("[calibrate] poison_threshold = %.1f  (mean - %.1f*std)" % (poison_threshold, POISON_STD_K))
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
def detect_edges_obstacles(frame_bgr):
    h, w = frame_bgr.shape[:2]
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    #bilateral filter
    gray = cv2.bilateralFilter(gray, 9, 30, 30)

    #canny edge detection
    edges = cv2.Canny(gray, EDGE_CANNY_LOW, EDGE_CANNY_HIGH)

    #scan for obstacle edges
    step_size = 8
    edge_array = []

    col_w = w // N_COLS
    for col_idx in range(N_COLS):
        col_start = col_idx * col_w
        col_end = col_start + col_w

        #find first edge from bottom to top
        found_edge = False
        for y in range(h - 5, 0, -1):
            if edges[y, col_start:col_end].max() > 0:
                #found edge, calculate average position
                edge_positions = np.where(edges[y, col_start:col_end] > 0)[0]
                if len(edge_positions) > 0:
                    avg_x = col_start + int(np.mean(edge_positions))
                    edge_array.append((avg_x, y))
                    found_edge = True
                    break

        if not found_edge:
            edge_array.append((col_start + col_w // 2, 0))  # No edge found

    #Calculate edge obstacle ratio
    edge_ratio_per_col = []
    for col_idx in range(N_COLS):
        col_start = col_idx * col_w
        col_end = col_start + col_w
        edge_count = (edges[:, col_start:col_end] > 0).sum()
        ratio = edge_count / max(edges[:, col_start:col_end].size, 1)
        edge_ratio_per_col.append(round(ratio, 3))

    #danger edge
    danger_y = int(h * DANGER_LINE_Y)
    danger_edges = edges[danger_y:, col_w:2*col_w]
    danger_edge_ratio = (danger_edges > 0).sum() / max(danger_edges.size, 1) if danger_edges.size > 0 else 0.0

    return {
        'edges': edges,
        'edge_array': edge_array,
        'edge_ratios': edge_ratio_per_col,
        'danger_edge_ratio': round(danger_edge_ratio, 3),
    }

def detect_contours_obstacles(frame_bgr):
    h, w = frame_bgr.shape[:2]
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    #threshold to get binary image
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

    #find contour
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    obstacles = []
    contour_danger_ratio = 0.0

    col_w = w // N_COLS
    danger_y = int(h * DANGER_LINE_Y)

    for idx, contour in enumerate(contours):
        area = cv2.contourArea(contour)

        #filter by minimun area
        if area < CONTOUR_MIN_AREA:
            continue

        #approximate contour shape
        arc_length = cv2.arcLength(contour, True)
        approx_contour = cv2.approxPolyDP(contour, CONTOUR_APPROX_EPS * arc_length, True)

        #get bounding rect
        x, y, w_rect, h_rect = cv2.boundingRect(contour)

        #check if contour is in danger zone
        if y + h_rect > danger_y and col_w < x + w_rect < 2 * col_w:
            contour_danger_ratio = max(contour_danger_ratio, area / (h_rect * w_rect + 1))

        obstacles.append({
            'x': x,
            'y': y,
            'w': w_rect,
            'h': h_rect,
            'area': area,
            'shape_points': len(approx_contour),
        })

    #normalize danger ratio
    contour_danger_ratio = min(1.0, contour_danger_ratio)

    return {
        'binary': binary,
        'contours': obstacles,
        'contour_count': len(obstacles),
        'danger_ratio': round(contour_danger_ratio, 3),
    }

def detect_floor(frame_bgr):
    h, w = frame_bgr.shape[:2]
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    #histogram floor
    sx1, sx2 = int(w * 0.30), int(w * 0.70)
    sy1, sy2 = int(h * 0.85), h
    floor = hsv[sy1:sy2, sx1:sx2]

    #2.use calibrated histogram
    if THRESHOLDS['floor_hist'] is not None and THRESHOLDS['floor_hist'].size > 0:
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
    sample_bp = cv2.calcBackProject([floor], [0, 1], hist, [0, 180, 0, 256], scale=1)
    sample_bp_mean = float(np.mean(sample_bp))
    poison_thresh = THRESHOLDS['poison_threshold']
    poisoned = False
    if poison_thresh > 0:
        poisoned = sample_bp_mean < poison_thresh

    return {
        'mask': mask_full,
        'ratios': ratios,
        'danger_ratio': danger_ratio,
        'poisoned': poisoned,
        'sample_bp_mean': sample_bp_mean,
        'sample_box': (sx1, sy1, sx2, sy2),
        'scan_box':   (rx1, ry1, rx2, ry2),
    }

def fuse_detections(floor_info, edge_info, contour_info):
    #get danger signals from all layers
    floor_danger = floor_info['danger_ratio']
    edge_danger = edge_info['danger_edge_ratio']
    contour_danger = contour_info['danger_ratio']

    #weighted fusion
    fused_danger = (
        WEIGHT_FLOOR * floor_danger +
        WEIGHT_EDGE * edge_danger +
        WEIGHT_CONTOUR * contour_danger
    )

    #enhanced floor ratios with edge information
    floor_ratios = floor_info['ratios']
    edge_ratios = edge_info['edge_ratios']
    contour_count_per_col = [0, 0, 0]

    col_w = 320 // N_COLS
    for obstacle in contour_info['contours']:
        col_idx = min(2, obstacle['x'] // col_w)
        contour_count_per_col[col_idx] += 1

    #fused ratio per column
    fused_ratios = []
    for i in range(N_COLS):
        fused_r = (
            WEIGHT_FLOOR * floor_ratios[i] +
            WEIGHT_EDGE * edge_ratios[i] +
            WEIGHT_CONTOUR * min(1.0, contour_count_per_col[i] * 0.2)
        )
        fused_ratios.append(round(fused_r, 3))

    return {
        'fused_danger': round(fused_danger, 3),
        'fused_ratios': fused_ratios,
        'floor_danger': round(floor_danger, 3),
        'edge_danger': round(edge_danger, 3),
        'contour_danger': round(contour_danger, 3),
    }

#think
def decide_advanced(frame_bgr):
    #detect from all layers
    floor_info = detect_floor(frame_bgr)
    edge_info = detect_edges_obstacles(frame_bgr)
    contour_info = detect_contours_obstacles(frame_bgr)

    #fuse detections
    fusion = fuse_detections(floor_info, edge_info, contour_info)
    fused_danger = fusion['fused_danger']
    ratios = fusion['fused_ratios']

    #check danger
    if floor_info['poisoned']:
        cmd = 'x'
        reason = "POISONED: NOT THE FLOOR -> escape (bp=%.0f < %.0f)" % (floor_info['sample_bp_mean'], THRESHOLDS['poison_threshold'])
    elif fused_danger < THRESHOLDS['danger']:
        cmd = 'x'
        reason = "DANGER FUSED: %.2f < %.2f (F:%.2f|E:%.2f|C:%.2f)" % (
            fused_danger, THRESHOLDS['danger'],
            fusion['floor_danger'], fusion['edge_danger'], fusion['contour_danger'])
    else:
        cmd = _steer_from_ratios(ratios)
        reason = "FLOOR L/C/R = %s | fused_danger = %.2f | contours = %d" % (ratios, fused_danger, contour_info['contour_count'])

    result_info = {
        'cmd': cmd,
        'reason': reason,
        'floor_info': floor_info,
        'edge_info': edge_info,
        'contour_info': contour_info,
        'fusion': fusion,
        'ratios': ratios,
    }
    return result_info


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
def build_view_advanced(frame_bgr, decision_info, fps):
    h, w = frame_bgr.shape[:2]
    original = frame_bgr.copy()
    result = frame_bgr.copy()

    floor_info = decision_info['floor_info']
    edge_info = decision_info['edge_info']
    contour_info = decision_info['contour_info']
    fusion = decision_info['fusion']

    #paint green - floor
    mask = floor_info['mask']
    green_overlay = np.zeros_like(result)
    green_overlay[mask > 0] = (0, 200, 0)
    result = cv2.addWeighted(result, 1.0, green_overlay, 0.35, 0)

    #paint blue - edge
    edges = edge_info['edges']
    blue_overlay = np.zeros_like(result)
    blue_overlay[edges > 0] = (255, 100, 0)
    result = cv2.addWeighted(result, 1.0, blue_overlay, 0.20, 0)

    #paint red - contour
    for obstacle in contour_info['contours']:
        x, y, w_rect, h_rect = obstacle['x'], obstacle['y'], obstacle['w'], obstacle['h']
        cv2.rectangle(result, (x, y), (x + w_rect, y + h_rect), (0, 0, 255), 2)

    #floor area
    sx1, sy1, sx2, sy2 = floor_info['sample_box']
    poisoned = floor_info.get('poisoned', False)
    sample_color = (0, 0, 255) if poisoned else (0, 255, 255)
    sample_label = "POISONED!" if poisoned else "Floor color"
    cv2.rectangle(result, (sx1, sy1), (sx2, sy2), sample_color, 2 if poisoned else 1)
    cv2.putText(result, sample_label, (sx1 + 2, sy1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)

    #draw lines and ratios
    ratios = decision_info['ratios']
    col_w = w // N_COLS
    for i in range(N_COLS):
        x0 = i * col_w
        cv2.line(result, (x0, 0), (x0, h), (255, 200, 0), 1)
        free = ratios[i] >= THRESHOLDS['free']
        color = (0, 220, 0) if free else (0, 0, 255)
        cv2.putText(result, "%.2f" % ratios[i], (x0 + 6, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    #danger line
    dy = int(h * DANGER_LINE_Y)
    danger_triggered = fusion['fused_danger'] < THRESHOLDS['danger']
    line_color = (0, 0, 255) if danger_triggered else (0, 200, 255)
    cv2.line(result, (0, dy), (w, dy), line_color, 2)
    danger_text = "DANGER (%.2f)" % fusion['fused_danger']
    cv2.putText(result, danger_text, (w - 160, dy - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, line_color, 1)


    #label
    _label(original, "VIDEO")
    _label(result,   "RESULT floor+edge+contour")

    #info
    info_h = 110
    canvas = np.zeros((h + info_h, w * 2, 3), dtype=np.uint8)
    canvas[0:h, 0:w] = original
    canvas[0:h, w:w * 2] = result

    cmd = decision_info['cmd']
    cmd_color = {'w': (0, 220, 0), 'x': (0, 0, 255), 'a': (0, 200, 255), 'd': (0, 200, 255)}.get(cmd, (255, 255, 255))
    cv2.putText(canvas, "CMD: %s" % cmd, (12, h + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.85, cmd_color, 2)
    reason_short = decision_info['reason'][:90]
    cv2.putText(canvas, reason_short, (12, h + 62), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1)
    cv2.putText(canvas, "FPS: %.1f | Contours: %d" % (fps, contour_info['contour_count']), (w * 2 - 200, h + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
    cv2.putText(canvas, "F:%.2f E:%.2f C:%.2f | ACT=%.2fs" % (fusion['floor_danger'], fusion['edge_danger'], fusion['contour_danger'], ACT_DURATION), (w * 2 - 250, h + 62), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (140, 140, 140), 1)

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
        print("[run] MANUAL MODE: floor=%d free=%.2f danger=%.2f edge=%.3f" % (THRESHOLDS['floor'], THRESHOLDS['free'], THRESHOLDS['danger'], THRESHOLDS['edge_threshold']))

    motor = MotorController(USE_ARDUINO, ARDUINO_PORT, ARDUINO_BAUD)

    writer = None
    if SAVE_VIDEO:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out_w  = FRAME_W * 2 * DISPLAY_SCALE
        out_h  = (FRAME_H + 90) * DISPLAY_SCALE
        writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, 10.0, (FRAME_W * 2, FRAME_H + 110))

    if STREAM_ENABLE:
        start_stream_server(STREAM_PORT)
        print("[stream]   http://<ip-address>:%d" % STREAM_PORT)
        print("[stream]ip: hostname -I)")

    print("[run] SATRT")

    escape_turn_dir = 'a'

    try:
        while True:
            #sense
            ok, frame = cap.read()
            if not ok or frame is None:
                print("[ERROR] CANNOT READ FRAME")
                break

            #think - enhanced
            t0 = time.time()
            decision_info = decide_advanced(frame)
            proc_time = time.time() - t0
            fps = 1.0 / max(proc_time, 1e-6)
            cmd = decision_info['cmd']

            #update
            view = build_view_advanced(frame, decision_info, fps)
            if STREAM_ENABLE:
                _frame_buffer.update(view)
            if writer is not None:
                writer.write(view)

            #act
            if cmd == 'x':
                escape_reason = "POISONED" if decision_info['floor_info'].get('poisoned') else "DANGER"
                print("[escape] %s -- backward %.1fs and turn %s %.1fs" % (escape_reason, ESCAPE_BACK_S, escape_turn_dir, ESCAPE_TURN_S))
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
