import cv2
import threading
import time
from flask import Flask, Response

app = Flask(__name__)
state = {"front": None, "right": None}
lock = threading.Lock()

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

def gen_stream(cam_key):
	while True:
		with lock:
			frame = state[cam_key]
		if frame is None:
			time.sleep(0.1)
		else:
			_, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
			yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
			time.sleep(0.033)

@app.route('/front')
def front():
	return Response(gen_stream('front'),
			mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/right')
def right():
	return Response(gen_stream('right'),
			mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
	return '''
	<html><body style="background:#111;display:flex;gap:10px;padding:10px">
	<div>
		<p style="color:white">Camera FRONT</p>
		<img src="/front" width="480">
	</div>
	<div>
		<p style="color:white">Camera RIGHT</p>
		<img src="/right" width="480">
	</div>
	</body></html>
	'''

if __name__ == '__main__':
	t = threading.Thread(target=read_cameras, daemon=True)
	t.start()
	print("Warming up cameras...")
	time.sleep(4)
	print("Starting server")
	app.run(host='0.0.0.0', port=5000, threaded=True) 
