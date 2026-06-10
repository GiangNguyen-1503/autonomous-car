import cv2
import time

bg_front = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=80, detectShadows=False)

state = {"front": None}
import threading

def read_camera():
	cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
	cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
	cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
	while True:
		ret, f = cap.read()
		if ret: state["front"] = f.copy()
		time.sleep(0.05)

def detect_area(frame):
	if frame is None: return 0
	h, w = frame.shape[:2]
	roi = frame[h//2:h, :]
	gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
	blur = cv2.GaussianBlur(gray, (7, 7), 0)
	edges = cv2.Canny(blur, 30, 90)
	contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	return max([cv2.contourArea(c) for c in contours], default=0)

threading.Thread(target=read_camera, daemon=True).start()
time.sleep(3)

print("Test Canny detection:")
print("Empty area | Nearby hand area | Wall contact area")
print("-" * 50)

while True:
	area = detect_area(state["front"])
	print(f"Area = {int(area)}", end ="\r")
	time.sleep(0.1)
