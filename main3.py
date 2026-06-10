import cv2
import numpy as np
import serial
import time
from collections import deque
import logging

SERIAL_PORT = '/dev/ttyACM0'
BAUD_RATE = 9600
SERIAL_TIMEOUT = 1

FRONT_CAMERA_ID = 0
RIGHT_CAMERA_ID = 2
FRAME_WIDTH = 320
FRAME_HEIGHT = 240

MOG2_VAR_THRESHOLD = 100
MOG2_HISTORY = 300
MOG2_DETECT_SHADOWS = True

MIN_FOREGROUND_PIXELS = 500
CENTROID_HISTORY_SIZE = 5
BLOCK_COUNT_THRESHOLD = 3

MORPH_KERNEL_SIZE = (5, 5)

#logging
logging.basicConfig(
	level=logging.INFO,
	format='%(asctime)s - %(levelname)s - %(message)s',
	handlers=[logging.FileHandler('/tmp/autonomous_car.log'), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

#serial
class SerialCommunicator:
	def __init__(self, port=SERIAL_PORT, baudrate=BAUD_RATE, timeout=SERIAL_TIMEOUT):
		self.port = port
		self.baudrate = baudrate
		self.timeout = timeout
		self.ser = None
		self.connect()
	def connect(self):
		try:
			self.ser = serial.Serial(self.port, self.baudrate, timeout=self.timeout)
			time.sleep(2)
			logger.info(f"Connected to {self.port} @ {self.baudrate} baud")
		except Exception as e:
			logger.error(f"Failed to connect to {self.port}: {e}")
			self.ser = None
	def send_command(self, command):
		if self.ser is None:
			logger.warning("Serial not connected, skipping command send")
			return False
		try:
			self.ser.write(command.encode())
			logger.debug(f"Sent: {command}")
			return True
		except Exception as e:
			logger.error(f"Failed to send command '{command}': {e}")
			return False
	def close(self):
		if self.ser:
			self.ser.close()
			logger.info("Serial connection closed")

#detection
class ObstacleDetector:
	def __init__(self):
        # MOG2 (Background Subtraction)
		self.mog2 = cv2.createBackgroundSubtractorMOG2(detectShadows=MOG2_DETECT_SHADOWS, varThreshold=MOG2_VAR_THRESHOLD, history=MOG2_HISTORY)
		self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, MORPH_KERNEL_SIZE)
        # HOG (Human Detection - only if foreground detected)
		self.hog = cv2.HOGDescriptor()
		self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector()) 

		logger.info("Obstacle detector initialized")

	def detect(self, frame):
		# Step 1: MOG2 Background Subtraction
		fg = self.mog2.apply(frame)
		# Step 2: Morphological Cleanup
		fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, self.kernel, iterations=1)
		fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, self.kernel, iterations=1)
		# Step 3: Calculate Histogram (centroid)
		hist = np.sum(fg, axis=0)
		total_pixels = np.sum(fg)
		is_blocked = total_pixels > MIN_FOREGROUND_PIXELS
		obstacle_type = 'NONE'
		centroid_x = None

		if is_blocked:
			centroid_x = np.argmax(hist)
			obstacle_type = 'STATIC'
			# Step 4: HOG for Person Detection (only if foreground)
			try:
				detections, weights = self.hog.detectMultiScale(frame, winStride=(8, 8), padding=(16, 16), scale=1.05, hitThreshold=2)
				people = detections if len(detections) > 0 else []	
				if len(people) > 0:
					person_x_positions = [x + w//2 for (x, y, w, h) in people]
					centroid_x = int(np.mean(person_x_positions))
					obstacle_type = 'PERSON'
					logger.debug(f"Person detected at x={centroid_x}")
			except Exception as e:
				logger.warning(f"HOG detection failed: {e}")
				people = []
		return is_blocked, obstacle_type, centroid_x, total_pixels
	def get_foreground_display(self, frame):
		return self.mog2.apply(frame)

#avoidance
class AvoidanceController:
	def __init__(self):
		self.centroid_history = deque(maxlen=CENTROID_HISTORY_SIZE)
		self.block_count = 0
		self.tried_left = False
		logger.info("Avoidance controller initialized")

	def decide_action(self, is_blocked, centroid_x):
		if is_blocked:
			self.block_count += 1
			if centroid_x is not None:
				self.centroid_history.append(centroid_x)
		else:
			self.block_count = 0
			self.centroid_history.clear()
			self.tried_left = False
			return 'F', 'Free space ahead'

		if self.block_count < BLOCK_COUNT_THRESHOLD:
			return 'F', f'Transient detection ({self.block_count}/{BLOCK_COUNT_THRESHOLD})'

		#histogram direction logic
		if len(self.centroid_history) > 0:
			avg_centroid = np.mean(self.centroid_history)
		else:
			avg_centroid = centroid_x if centroid_x else FRAME_WIDTH // 2

		#devide frame into regions
		left_boundary = FRAME_WIDTH // 3
		right_boundary = 2 * FRAME_WIDTH // 3

		#tried_left
		if avg_centroid < left_boundary:
			command = 'R'
			reason = f'Obstacle left (x={avg_centroid:.0f}), turning right'
			self.tried_left = False
		elif avg_centroid > right_boundary:
			command = 'L'
			reason = f'Obstacle right (x={avg_centroid:.0f}), turning left'
			self.tried_left = True
		else:
			if not self.tried_left:
				command = 'L'
				reason = f'Obstacle center (x={avg_centroid:.0f}), trying left'
				self.tried_left = True
			else:
				command = 'R'
				reason = f'Obstacle center (x={avg_centroid:.0f}), trying right'
				self.tried_left = False

		return command, reason

#main
class AutonomousVehicle:
	def __init__(self):
		self.detector = ObstacleDetector()
		self.controller = AvoidanceController()
		self.serial = SerialCommunicator()

		self.cap_front = cv2.VideoCapture(FRONT_CAMERA_ID)
		self._setup_camera(self.cap_front)

		self.cap_right = None
		try:
			self.cap_right = cv2.VideoCapture(RIGHT_CAMERA_ID)
			self._setup_camera(self.cap_right)
			logger.info("Right-side camera initialized")
		except Exception as e:
			logger.warning(f"Right-side camera not available: {e}")
		self.frame_count = 0
		self.fps_timer = time.time()
		self.fps = 0
		logger.info("Autonomous vehicle initialized")
	def _setup_camera(self, cap):
		cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
		cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
		cap.set(cv2.CAP_PROP_FPS, 30)
		cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
	def process_frame(self, frame):
		# Detect obstacles
		is_blocked, obstacle_type, centroid_x, fg_pixels = self.detector.detect(frame)
		# Decide action
		command, reason = self.controller.decide_action(is_blocked, centroid_x)
		# Send to Arduino
		self.serial.send_command(command)
		annotated = self._annotate_frame(frame, is_blocked=is_blocked, obstacle_type=obstacle_type, centroid_x=centroid_x, fg_pixels=fg_pixels, command=command, reason=reason)
		return annotated, command
	def _annotate_frame(self, frame, is_blocked, obstacle_type, centroid_x, fg_pixels, command, reason):
		annotated = frame.copy()
		status_color = (0, 0, 255) if is_blocked else (0, 255, 0)
		cv2.rectangle(annotated, (0, 0), (FRAME_WIDTH, 40), status_color, -1)
		cv2.putText(annotated, f"{obstacle_type} | {command} | {reason}", (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
		if centroid_x is not None:
			cv2.circle(annotated, (centroid_x, FRAME_HEIGHT // 2), 5, (0, 255, 255), -1)
			cv2.line(annotated, (centroid_x, 0), (centroid_x, FRAME_HEIGHT), (0, 255, 255), 1)

		#draw frame
		left_boundary = FRAME_WIDTH // 3
		right_boundary = 2 * FRAME_WIDTH // 3
		cv2.line(annotated, (left_boundary, 0), (left_boundary, FRAME_HEIGHT), (100, 100, 100), 1)
		cv2.line(annotated, (right_boundary, 0), (right_boundary, FRAME_HEIGHT), (100, 100, 100), 1)

		cv2.putText(annotated, f"FPS: {self.fps:.1f}", (FRAME_WIDTH - 60, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
		return annotated
	def update_fps(self):
		self.frame_count += 1
		elapsed = time.time() - self.fps_timer
		if elapsed >= 1.0:
			self.fps = self.frame_count / elapsed
			self.frame_count = 0
			self.fps_timer = time.time()
			logger.info(f"FPS: {self.fps:.1f}")
	def run(self):
		logger.info("Starting main loop...")
		try:
			while True:
				ret, frame = self.cap_front.read()
				if not ret:
					logger.error("Failed to read frame from front camera")
					break

				annotated, command = self.process_frame(frame)
				self.update_fps()
				#cv2.imshow('Front Camera', annotated)
				#if self.cap_right:
					#ret_right, frame_right = self.cap_right.read()
					#if ret_right:
						#cv2.imshow('Right Camera', frame_right)
				#key control
				key = cv2.waitKey(1) & 0xFF
				if key == ord('q'):
					logger.info("Quit signal received")
					break
				elif key == ord('s'):
					logger.warning("Emergency stop!")
					self.serial.send_command('S')
					time.sleep(0.5)
		except KeyboardInterrupt:
			logger.info("Interrupted by user")
		except Exception as e:
			logger.error(f"Unexpected error: {e}", exc_info=True)
		finally:
			self.cleanup()
	def cleanup(self):
		logger.info("Cleaning up...")
		self.serial.send_command('S')
		self.serial.close()
		self.cap_front.release()
		if self.cap_right:
			self.cap_right.release()
		cv2.destroyAllWindows()
		logger.info("Cleanup complete")

#entry
if __name__ == '__main__':
	logger.info("=" * 80)
	logger.info("Autonomous Obstacle-Avoidance Vehicle")
	logger.info("RPi4 4B | Arduino Uno | L293D Motor Shield")
	logger.info("=" * 80)
	vehicle = AutonomousVehicle()
	vehicle.run()
