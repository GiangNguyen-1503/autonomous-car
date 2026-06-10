import cv2
import threading
import time

state = {"front": None, "right": None}

def read_cameras():
    cap_f = cv2.VideoCapture(0, cv2.CAP_V4L2)
    cap_r = cv2.VideoCapture(2, cv2.CAP_V4L2)
    cap_f.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap_f.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    cap_r.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap_r.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    while True:
        ret_f, f = cap_f.read()
        ret_r, r = cap_r.read()
        if ret_f:
            state["front"] = f
        if ret_r:
            state["right"] = r

def detect_obstacle(frame, min_area=800):
    if frame is None:
        return False
    h, w = frame.shape[:2]
    roi = frame[int(h * 0.3):h, :]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 100)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) > min_area:
            return True
    return False

if __name__ == "__main__":
    t = threading.Thread(target=read_cameras, daemon=True)
    t.start()
    time.sleep(2)

    print("Detection running. Press Ctrl+C to stop.")
    try:
        while True:
            front_blocked = detect_obstacle(state["front"])
            right_blocked = detect_obstacle(state["right"], min_area=500)
            print(f"Front blocked: {front_blocked} | Right clear: {not right_blocked}")
            time.sleep(0.2)
    except KeyboardInterrupt:
        print("Stopped.")
