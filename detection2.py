import cv2
import threading
import time

# Background subtractor
bg_front = cv2.createBackgroundSubtractorMOG2(
    history=100, varThreshold=40, detectShadows=False)
bg_right = cv2.createBackgroundSubtractorMOG2(
    history=100, varThreshold=40, detectShadows=False)

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

def detect_obstacle(frame, bg_subtractor, min_area=1500):
    if frame is None:
        return False
    # Apply background subtraction
    fg_mask = bg_subtractor.apply(frame)
    # Remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
    fg_mask = cv2.dilate(fg_mask, kernel, iterations=2)
    # Find contours in foreground mask
    contours, _ = cv2.findContours(
        fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) > min_area:
            return True
    return False

if __name__ == "__main__":
    t = threading.Thread(target=read_cameras, daemon=True)
    t.start()
    time.sleep(3)  # Wait longer for background to learn

    print("Detection running. Keep camera still for 3 seconds first!")
    print("Press Ctrl+C to stop.")
    try:
        while True:
            front_blocked = detect_obstacle(
                state["front"], bg_front)
            right_blocked = detect_obstacle(
                state["right"], bg_right, min_area=1000)
            print(f"Front: {'BLOCKED' if front_blocked else 'clear'} | "
                  f"Right: {'BLOCKED' if right_blocked else 'clear'}")
            time.sleep(0.15)
    except KeyboardInterrupt:
        print("Stopped.")
