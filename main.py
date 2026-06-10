import cv2
import serial
import threading
import time
import numpy as np

block_count = 0
block_THRESHOLD = 3

ser = serial.Serial('/dev/ttyACM0', 9600, timeout=1)
time.sleep(2)

def send(cmd):
	ser.write(cmd.encode())

bg_front = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=80, detectShadows=False)
bg_right = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=80, detectShadows=False)

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

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

def has_foreground(frame, bg_sub, min_area=1500):
	if frame is None:
		return False
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
	fg = bg_sub.apply(frame)
	fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel)
	fg = cv2.dilate(fg, kernel, iterations=2)
	contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	for cnt in contours:
		if cv2.contourArea(cnt) > min_area:
			return True
	return False

def detect_person(frame):
	if frame is None:
		return False
	resized = cv2.resize(frame, (320, 240))
	boxes, _ = hog.detectMultiScale(resized, winStride=(16, 16), padding=(8, 8), scale=1.1)
	return len(boxes) > 0

tried_left = False

def decide(front_blocked, right_clear):
	global tried_left

	if not front_blocked:
		tried_left = False
		return 'F'  # Forward

	if not tried_left:
		tried_left = True
		return 'L'  # Try left first

	if right_clear:
		tried_left = False
		return 'R'  # Fallback right

	# All block
	tried_left = False
	return 'P'  # Pivot turn

def pivot():
	send('L')
	time.sleep(1.5)
	send('S')

def main():
	global block_count
	global block_THRESHOLD
	cam_thread = threading.Thread(target=read_cameras, daemon=True)
	cam_thread.start()

	print("Warming up cameras and background model...")
	time.sleep(3)
	print("Starting autonomous navigation!")

	try:
		while True:
			f = state["front"]
			r = state["right"]
			
			raw_blocked = has_foreground(f, bg_front)
			if raw_blocked:
				block_count += 1
			else:
				block_count = 0
			front_blocked = block_count >= block_THRESHOLD 
			right_clear   = not has_foreground(r, bg_right, min_area=1000)

            		# Run HOG only when front blocked
			if front_blocked:
				is_person = detect_person(f)
				label = "PERSON" if is_person else "OBJECT"
				print(f"BLOCKED ({label}) | Right: {'clear' if right_clear else 'blocked'}")
			else:
				print("Clear — moving forward")
			
			cmd = decide(front_blocked, right_clear)

			if cmd == 'P':
				pivot()
			else:
				send(cmd)

			time.sleep(0.2)

	except KeyboardInterrupt:
		send('S')
		ser.close()
		print("Stopped.")

if __name__ == "__main__":
	main()
