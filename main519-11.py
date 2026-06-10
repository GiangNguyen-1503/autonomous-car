import time
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

#  main_freespace.py -- Free-Space Navigation (BFR pseudo code)
#
#  TRIET LY MOI: Khong "ne vat can", ma "di ve vung thoang nhat".
#
#  Pipeline:
#   1. Canny edge detection (loc bilateral truoc)
#   2. Column scan: voi moi cot, tim edge thap nhat tu day len
#      -> EdgeArray = "skyline" cua vat can
#      -> y cao = vat can gan, y = 0 = duong thong
#   3. Chia EdgeArray thanh 3 vung: LEFT / FORWARD / RIGHT
#      Tinh trung binh y moi vung -> c_left, c_fwd, c_right
#   4. Quyet dinh:
#      - Neu c_fwd > danger_y (vat can rat gan phia truoc)
#        -> chon vung co y NHO NHAT (thoang nhat)
#        -> quay LEFT hoac RIGHT theo vi tri vung do
#      - Nguoc lai: di FORWARD, lech nhe ve vung thoang hon
#
#   5. Lop phu tro: floor color back-projection xac nhan
#      Neu Canny bao "thoang" nhung floor color khong xac nhan
#      -> giam tin cay (an toan).
#
#  FSM 2 trang thai van giu, tranh oscillation L/R.

# --- CAU HINH chung ---
CAM_INDEX     = 0
FRAME_W       = 320
FRAME_H       = 240
DISPLAY_SCALE = 2

# --- Canny + column scan ---
STEP_SIZE     = 8         # khoang cach giua cac cot quet (pixel)
CANNY_LOW     = 50
CANNY_HIGH    = 100
ROI_TOP_RATIO = 0.35      # bo phan tren 35% (tran nha, tuong xa)
SAFE_BOTTOM   = 5         # bo 5 pixel day (camera distortion)

# --- Free-space decision thresholds ---
DANGER_Y_RATIO   = 0.55   # neu c_fwd > 55% chieu cao ROI -> vat can rat gan
STEER_MARGIN_PX  = 15     # chenh lech y giua 2 vung > 15px moi lech huong
N_REGIONS        = 3      # L / F / R

# --- BLIND WALL detection (fix loi "tuong trong") ---
# Khi ca 3 vung deu ~0 VA tong so edge rat it -> camera nhin vao be mat
# phang vo dac trung (tuong) -> KHONG tin la thoang -> re trai do duong.
# Phan biet voi "san that su rong" (co vai edge xa o phia tren).
BLIND_C_MAX        = 6      # ca 3 vung deu < 6px coi nhu "phang"
BLIND_EDGE_RATIO   = 0.015  # ti le pixel edge trong ROI < 1.5% -> vo dac trung
BLIND_TURN_DIR     = 'a'    # huong do duong khi gap tuong mu (trai)

# --- Floor color phu tro ---
USE_FLOOR_AUX        = True
FLOOR_CONFIRM_RATIO  = 0.30   # neu floor_ratio < 30% -> giam tin cay "thoang"

# --- FSM ---
CONFIRM_FRAMES = 2        # so khung DANGER lien tiep -> chuyen sang AVOIDING
EXIT_CONFIRM   = 3        # so khung "thoang" lien tiep -> thoat AVOIDING
WARM_UP_FRAMES = 2        # sau khi thoat, khong trigger DANGER trong N khung

# --- Timing ---
ACT_DURATION    = 0.3     # nhip vong lap thong thuong
TURN_NUDGE_S    = 0.3     # moi nhip xoay trong AVOIDING
ENTRY_BACK_S    = 0.35    # lui 1 lan luc vao AVOIDING

# --- Hardware ---
USE_ARDUINO  = True
ARDUINO_PORT = '/dev/ttyACM0'
ARDUINO_BAUD = 9600

# --- Save & stream ---
SAVE_VIDEO    = True
OUTPUT_PATH   = 'run_freespace.avi'
STREAM_ENABLE = True
STREAM_PORT   = 8000


#  MJPEG STREAM (giu nguyen tu ban cu)
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


#  MOTOR CONTROLLER (giu nguyen)
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


#  CANNY + COLUMN SCAN -- trai tim cua thuat toan moi
def scan_edges(frame_bgr):
    """
    Quet edge tu day len de tim "skyline" vat can.
    Tra ve dict gom:
      edge_array : list[(x, y_local)]  -- y_local trong toa do ROI
      roi_h      : chieu cao ROI
      roi_top    : offset top cua ROI trong frame goc
      edges      : anh edge (de debug)
      c_left, c_fwd, c_right : trung binh y_local cua 3 vung
                               (cao = vat can gan, 0 = thoang)
    """
    h, w = frame_bgr.shape[:2]

    # ROI: bo phan tren (tran nha, tuong xa khong dang lo)
    roi_top = int(h * ROI_TOP_RATIO)
    roi = frame_bgr[roi_top:h, :]
    roi_h, roi_w = roi.shape[:2]

    # Bilateral filter giu canh sac net, lam muot vung dong nhat
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 9, 30, 30)

    edges = cv2.Canny(gray, CANNY_LOW, CANNY_HIGH)

    # Tong ti le pixel edge trong ROI -> phan biet tuong mu vs san that
    edge_ratio = float((edges > 0).sum()) / max(edges.size, 1)

    # Column scan: voi moi cot, quet tu day len, tim pixel trang dau tien
    edge_array = []
    for x in range(0, roi_w, STEP_SIZE):
        found = False
        for y in range(roi_h - SAFE_BOTTOM, 0, -1):
            if edges[y, x] == 255:
                # y_local: do cao tu day ROI (0 = day, lon = cao)
                y_local = roi_h - 1 - y
                edge_array.append((x, y_local))
                found = True
                break
        if not found:
            edge_array.append((x, 0))   # khong gap edge -> thoang

    # Chia 3 vung
    n = len(edge_array)
    if n < N_REGIONS:
        c_left = c_fwd = c_right = 0
    else:
        chunk = n // N_REGIONS
        left_ys  = [p[1] for p in edge_array[0:chunk]]
        fwd_ys   = [p[1] for p in edge_array[chunk:2*chunk]]
        right_ys = [p[1] for p in edge_array[2*chunk:]]
        c_left  = float(np.mean(left_ys))  if left_ys  else 0
        c_fwd   = float(np.mean(fwd_ys))   if fwd_ys   else 0
        c_right = float(np.mean(right_ys)) if right_ys else 0

    return {
        'edge_array': edge_array,
        'roi_h':      roi_h,
        'roi_w':      roi_w,
        'roi_top':    roi_top,
        'edges':      edges,
        'edge_ratio': edge_ratio,
        'c_left':     c_left,
        'c_fwd':      c_fwd,
        'c_right':    c_right,
    }


#  FLOOR COLOR AUX -- lop phu tro xac nhan
def floor_color_aux(frame_bgr, roi_top):
    """
    Tra ve floor_ratio: ti le pixel "giong san" trong vung phia truoc xe.
    Cao = sàn rong, thấp = co the bi che boi vat can.
    """
    h, w = frame_bgr.shape[:2]
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    hsv = cv2.bilateralFilter(hsv, 5, 50, 50)

    # Sample sàn ngay duoi xe (chac chan la san)
    sx1, sx2 = int(w * 0.30), int(w * 0.70)
    sy1, sy2 = int(h * 0.85), h
    floor = hsv[sy1:sy2, sx1:sx2]

    if floor.size == 0:
        return 0.0

    hist = cv2.calcHist([floor], [0, 1], None, [30, 32], [0, 180, 0, 256])
    cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)

    # Back-project vào vùng phía trước (giữa ROI và sàn mẫu)
    region = hsv[roi_top:sy1, :]
    if region.size == 0:
        return 0.0
    bp = cv2.calcBackProject([region], [0, 1], hist, [0, 180, 0, 256], scale=1)
    bp = cv2.medianBlur(bp, 5)
    _, bp_bin = cv2.threshold(bp, 40, 255, cv2.THRESH_BINARY)

    return float((bp_bin > 0).sum()) / max(bp_bin.size, 1)

#fsm
class FreeSpaceFSM:
    CRUISE   = "CRUISE"
    AVOIDING = "AVOIDING"

    def __init__(self):
        self.state = self.CRUISE
        self.locked_turn_dir = None
        self.danger_streak   = 0
        self.clear_streak    = 0
        self.just_entered    = False
        self.warm_up         = 0

    def _is_danger(self, info, roi_h):
        """Vat can rat gan phia truoc."""
        danger_y = DANGER_Y_RATIO * roi_h
        return info['c_fwd'] > danger_y

    def _is_blind_wall(self, info):
        """
        Tuong mu: ca 3 vung gan nhu phang (c ~ 0) VA rat it edge trong ROI.
        -> camera dang nhin vao be mat phang vo dac trung (tuong).
        -> KHONG duoc tin la 'thoang'.
        """
        c_l, c_f, c_r = info['c_left'], info['c_fwd'], info['c_right']
        all_flat = (c_l < BLIND_C_MAX and c_f < BLIND_C_MAX and c_r < BLIND_C_MAX)
        few_edges = info['edge_ratio'] < BLIND_EDGE_RATIO
        return all_flat and few_edges

    def _is_clear(self, info, roi_h):
        """Phia truoc thoang (c_fwd thap) VA khong phai tuong mu."""
        if self._is_blind_wall(info):
            return False   # tuong mu khong phai 'thoang'
        danger_y = DANGER_Y_RATIO * roi_h
        # Yeu cau c_fwd phai NHO HON nhieu so voi nguong danger
        return info['c_fwd'] < danger_y * 0.6

    def _best_direction(self, info):
        """
        Chon vung co y NHO NHAT (thoang nhat).
        Tra ve 'a' (trai), 'd' (phai), hoac None (giua thoang nhat -> di thang).
        """
        c_l, c_f, c_r = info['c_left'], info['c_fwd'], info['c_right']
        if c_f <= c_l and c_f <= c_r:
            return None  # giua thoang nhat
        if c_l < c_r:
            return 'a'
        return 'd'

    def step(self, info, floor_ratio, roi_h):
        c_l, c_f, c_r = info['c_left'], info['c_fwd'], info['c_right']

        # === CRUISE ===
        if self.state == self.CRUISE:
            # Warm-up sau khi vua thoat AVOIDING
            if self.warm_up > 0:
                self.warm_up -= 1
                self.danger_streak = 0
                return ('w',
                        "WARM-UP %d/%d (c_fwd=%.0f)" % (self.warm_up, WARM_UP_FRAMES, c_f),
                        self.state)

            danger = self._is_danger(info, roi_h)

            # === BLIND WALL: ca 3 vung phang + it edge -> tuong mu ===
            # Day la fix cho loi "dung tuong nhung van di thang".
            # Khi gap tuong mu, re trai cham de do duong, KHONG di thang.
            if self._is_blind_wall(info):
                self.danger_streak = 0   # khong tinh la danger spike
                return (BLIND_TURN_DIR,
                        "BLIND WALL: flat surface, probing %s (edge=%.3f)" %
                        (BLIND_TURN_DIR, info['edge_ratio']),
                        self.state)

            # Floor aux: neu Canny bao thoang nhung floor color khong xac nhan
            # -> coi nhu khong tin cay, tang nguy co
            if USE_FLOOR_AUX and not danger and floor_ratio < FLOOR_CONFIRM_RATIO:
                # floor khong xac nhan -> kha nang co vat can mà Canny không thấy
                # (vd: tuong trang phang) -> nâng cảnh báo lên
                danger = True

            if danger:
                self.danger_streak += 1
                if self.danger_streak >= CONFIRM_FRAMES:
                    # Vao AVOIDING, khoa huong theo "best direction"
                    direction = self._best_direction(info)
                    if direction is None:
                        # Truong hop hiem: c_fwd lon nhung van la nho nhat
                        # -> uu tien trai
                        direction = 'a'
                    self.locked_turn_dir = direction
                    self.state = self.AVOIDING
                    self.just_entered = True
                    self.clear_streak = 0
                    self.danger_streak = 0
                    return ('x',
                            "ENTER AVOIDING, lock=%s (L=%.0f F=%.0f R=%.0f)" %
                            (direction, c_l, c_f, c_r),
                            self.state)
                # Chua du khung -> lech nhe theo huong tot hon
                if c_l < c_r - STEER_MARGIN_PX:
                    return ('a', "pre-avoid: nudge left (L=%.0f R=%.0f)" % (c_l, c_r), self.state)
                if c_r < c_l - STEER_MARGIN_PX:
                    return ('d', "pre-avoid: nudge right (L=%.0f R=%.0f)" % (c_l, c_r), self.state)
                return ('w', "pre-avoid: keep forward", self.state)

            # Khong danger -> di toi, lech ve vung thoang hon
            self.danger_streak = 0

            # Pseudo code BFR: di toi neu c_fwd ok, lech theo min(c)
            if c_l < c_r - STEER_MARGIN_PX and c_l < c_f:
                return ('a', "CRUISE: lean left (L=%.0f F=%.0f R=%.0f)" % (c_l, c_f, c_r), self.state)
            if c_r < c_l - STEER_MARGIN_PX and c_r < c_f:
                return ('d', "CRUISE: lean right (L=%.0f F=%.0f R=%.0f)" % (c_l, c_f, c_r), self.state)
            return ('w', "CRUISE: forward (L=%.0f F=%.0f R=%.0f)" % (c_l, c_f, c_r), self.state)

        # === AVOIDING ===
        if self.just_entered:
            self.just_entered = False
            return ('s', "AVOIDING: entry back-up", self.state)

        if self._is_clear(info, roi_h):
            self.clear_streak += 1
            if self.clear_streak >= EXIT_CONFIRM:
                old = self.locked_turn_dir
                self.state = self.CRUISE
                self.locked_turn_dir = None
                self.clear_streak = 0
                self.warm_up = WARM_UP_FRAMES
                return ('w', "EXIT AVOIDING (was=%s) -> CRUISE" % old, self.state)
            return ('w', "clear streak %d/%d (c_fwd=%.0f)" %
                    (self.clear_streak, EXIT_CONFIRM, c_f), self.state)

        self.clear_streak = 0
        return (self.locked_turn_dir,
                "AVOIDING: turning %s (L=%.0f F=%.0f R=%.0f)" %
                (self.locked_turn_dir, c_l, c_f, c_r),
                self.state)


#  BUILD VIEW -- visualize edge array + 3 vung + lenh
def build_view(frame_bgr, info, floor_ratio, cmd, reason, state, fps):
    h, w = frame_bgr.shape[:2]
    original = frame_bgr.copy()
    result = frame_bgr.copy()

    roi_top = info['roi_top']
    roi_h   = info['roi_h']
    roi_w   = info['roi_w']

    # Ve edge array len anh result (chuyen y_local ve toa do anh goc)
    edge_array = info['edge_array']
    pts_img = []
    for (x, y_local) in edge_array:
        y_img = (roi_top + roi_h - 1) - y_local
        pts_img.append((x, y_img))

    # Ve duong skyline
    for i in range(len(pts_img) - 1):
        cv2.line(result, pts_img[i], pts_img[i + 1], (0, 255, 0), 2)

    # Ve duong tu day toi moi diem (visual giong BFR goc)
    bottom_y = roi_top + roi_h - 1
    for i in range(0, len(pts_img), 2):  # thua hon de khong roi mat
        cv2.line(result, (pts_img[i][0], bottom_y), pts_img[i], (0, 180, 0), 1)

    # Ve duong ranh gioi 3 vung
    region_w = w // N_REGIONS
    for i in range(1, N_REGIONS):
        x = i * region_w
        cv2.line(result, (x, roi_top), (x, h), (255, 200, 0), 1)

    # Ve duong DANGER
    danger_y_local = int(DANGER_Y_RATIO * roi_h)
    danger_y_img = (roi_top + roi_h - 1) - danger_y_local
    cv2.line(result, (0, danger_y_img), (w, danger_y_img), (0, 0, 255), 1)
    cv2.putText(result, "DANGER", (w - 70, danger_y_img - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    # Hien thi c_left/c_fwd/c_right
    cs = [info['c_left'], info['c_fwd'], info['c_right']]
    labels = ['L', 'F', 'R']
    for i, (label, c) in enumerate(zip(labels, cs)):
        x0 = i * region_w + region_w // 2 - 20
        danger_y = DANGER_Y_RATIO * roi_h
        color = (0, 0, 255) if c > danger_y else (0, 220, 0)
        cv2.putText(result, "%s=%.0f" % (label, c), (x0, h - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Label panels
    _label(original, "VIDEO")
    _label(result, "RESULT (skyline + 3 regions)")

    # Canvas + thanh thong tin
    info_h = 90
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
    cv2.putText(canvas, "FPS: %.1f | floor_aux=%.2f | edge=%.3f" % (fps, floor_ratio, info['edge_ratio']),
                (w * 2 - 360, h + 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
    cv2.putText(canvas, "Free-space seeking | min(c) -> direction",
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


#  MAIN LOOP
def main():
    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    if not cap.isOpened():
        print("[ERROR] Cannot open camera"); sys.exit(1)

    print("[config] Free-space navigation: Canny + 3-region voting")
    print("[config] STEP_SIZE=%d  DANGER_Y_RATIO=%.2f  STEER_MARGIN=%dpx" %
          (STEP_SIZE, DANGER_Y_RATIO, STEER_MARGIN_PX))

    motor = MotorController(USE_ARDUINO, ARDUINO_PORT, ARDUINO_BAUD)
    fsm   = FreeSpaceFSM()

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

            # 1. Canny + column scan -> 3-region c values
            info = scan_edges(frame)

            # 2. Floor color aux (chi de xac nhan, khong quyet dinh chinh)
            if USE_FLOOR_AUX:
                floor_ratio = floor_color_aux(frame, info['roi_top'])
            else:
                floor_ratio = 1.0

            # 3. FSM quyet dinh
            cmd, reason, state = fsm.step(info, floor_ratio, info['roi_h'])

            proc_time = time.time() - t0
            fps = 1.0 / max(proc_time, 1e-6)

            # 4. Hien thi
            view = build_view(frame, info, floor_ratio, cmd, reason, state, fps)
            if STREAM_ENABLE: _frame_buffer.update(view)
            if writer is not None: writer.write(view)

            # 5. Gui lenh + sleep theo loai
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
