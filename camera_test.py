import cv2
import threading

def read_camera(index, name):
	cap = cv2.VideoCapture(index)
	if not cap.isOpened():
		print(f"{name}: Cannot open!!!")
		return
	ret, frame = cap.read()
	if ret:
		print(f"{name}: OK - shape {frame.shape}")
	else:
		print(f"{name}: Read Frame Failure!!!")
	cap.release()

t0 = threading.Thread(target=read_camera, args=(0, "Front Camera"))
t2 = threading.Thread(target=read_camera, args=(2, "Right Camera"))

t0.start()
t2.start()
t0.join()
t2.join()
