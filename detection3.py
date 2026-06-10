import cv2
import threading
import time

########
#HOG + BACKGROUND


# Background subtractors
bg_front = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40, detectShadows=False)
bg_right = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40, detectShadows=False)

# HOG person detector
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

def get_foreground(frame, bg_subtractor, min_area=1500):
    """Returns (has_object, fg_mask)"""
	if frame is None:
		return False, None
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
	fg = bg_subtractor.apply(frame)
	fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel)
	fg = cv2.dilate(fg, kernel, iterations=2)
	contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	for cnt in contours:
		if cv2.contourArea(cnt) > min_area:
			return True, fg
	return False, fg

def detect_person(frame):
    """Only called when foreground detected — saves CPU"""
	if frame is None:
		return False
	resized = cv2.resize(frame, (320, 240))
	boxes, _ = hog.detectMultiScale(resized, winStride=(16, 16), padding=(8, 8), scale=1.1)
	return len(boxes) > 0

def detect_obstacle(frame, bg_subtractor, min_area=1500):
    """
    Returns (obstacle_detected, is_person)
    Step 1: Background subtraction (fast)
    Step 2: HOG only if foreground found (slow but rare)
    """
	has_object, _ = get_foreground(frame, bg_subtractor, min_area)
	if not has_object:
		return False, False
    
	# Only run HOG when something is detected
	is_person = detect_person(frame)
	return True, is_person

def is_right_clear(frame):
	has_object, _ = get_foreground(frame, bg_right, min_area=1000)
	return not has_object

if __name__ == "__main__":
	t = threading.Thread(target=read_cameras, daemon=True)
	t.start()
	time.sleep(3)

	print("Detection running. Keep camera still for 3s first!")
	print("Press Ctrl+C to stop.")
	try:
		while True:
			f = state["front"]
			r = state["right"]
			blocked, is_person = detect_obstacle(f, bg_front)
			right_clear = is_right_clear(r)
			if blocked:
				label = "PERSON" if is_person else "OBJECT"
				print(f"Front: BLOCKED ({label}) | Right: {'clear' if right_clear else 'BLOCKED'}")
			else:
				print(f"Front: clear | Right: {'clear' if right_clear else 'BLOCKED'}")
			time.sleep(0.15)
	except KeyboardInterrupt:
		print("Stopped.")
