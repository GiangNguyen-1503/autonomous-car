import cv2
import serial
import threading
import time
import numpy as np

ser = serial.Serial("/dev/ttyACM0", 9600, timeout=1)
time.sleep(2)

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

bg_front = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=100, detectShadows=False)
bg_right = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=100, detectShadows=False)

state = {"front": None, "right": None}
lock = threading.Lock()
block_count = 0
BLOCK_THRESHOLD = 3
tried_left = False

def send(cmd):
	ser.write(cmd.encode())
	print(f"[CMD] {cmd}")

def read_cameras():
	cap_f = cv2.VideoCapture(0, cv2.CAP_V4L2)
	cap_r = cv2.VideoCapture(2, cv2.CAP_V4L2)
	for cap in [cap_f, cap_r]:
		cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
		cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
		cap.set(cv2.CAP_PROP_FPS, 15)
	while True:
		ret_f, f = cap_f.read()
		ret_r, r = cap_r.read()
		with lock:
			if ret_f: state["front"] = f.copy()
			if ret_r: state["right"] = r.copy()
		time.sleep(0.05)

def detect_obstacle(frame, bg_sub, min_area=1500):
	if frame is None:
		return False, 0
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
	fg = bg_sub.apply(frame)
	fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel)
	fg = cv2.dilate(fg, kernel, iterations=2)
	contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	max_area = max([cv2.contourArea(c) for c in contours], default=0)
	return max_area > min_area, max_area

def detect_person(frame):
	if frame is None:
		return False
	resized = cv2.resize(frame, (320, 240))
	boxes, _ = hog.detectMultiScale(resized, winStride=(16, 16), padding=(8, 8), scale=1.1)
	return len(boxes) > 0

def decide(front_blocked, right_clear):
	global tried_left
	if not front_blocked:
		tried_left = False
		return "F"
	if not tried_left:
		tried_left = True
		return "L"
	if right_clear:
		tried_left = False
		return "R"
	tried_left = False
	return "P"

def pivot():
	send("L")
	time.sleep(1.5)
	send("S")

def main():
	global block_count
	threading.Thread(target=read_cameras, daemon=True).start()
	print("Warming up cameras...")
	time.sleep(3)
	print("Starting autonomous navigation!")

	while True:
		with lock:
			f = state["front"].copy() if state["front"] is not None else None
			r = state["right"].copy() if state["right"] is not None else None
		front_blocked, front_area = detect_obstacle(f, bg_front, min_area=1500)
		right_blocked, right_area = detect_obstacle(r, bg_right, min_area=1000)
		right_clear = not right_blocked

		if front_blocked:
			block_count += 1
		else:
			block_count = 0
		confirmed = block_count >= BLOCK_THRESHOLD

		if confirmed:
			is_person = detect_person(f)
			label = "PERSON" if is_person else "OBJECT"
			print(f"[BLOCKED-{label}] front_area={int(front_area)} | "
				f"right_area={int(right_area)} | "
				f"right_clear={right_clear}")
		else:
			print(f"[CLEAR] front_area={int(front_area)} | "
				f"right_area={int(right_area)}")
		cmd = decide(confirmed, right_clear)

		if cmd == "P":
			pivot()
		else:
			send(cmd)

		time.sleep(0.2)

if __name__ == "__main__":
	try:
		main()
	except KeyboardInterrupt:
		send("S")
		ser.close()
		print("Stooped.")
