import serial
import time
import sys
import tty
import termios

ser = serial.Serial("/dev/ttyACM0", 9600, timeout=1)
time.sleep(2)
speed = 180;

print("Motor Manual Control")

def get_key():
	fd = sys.stdin.fileno()
	old_settings = termios.tcgetattr(fd)
	try:
		tty.setraw(fd)
		ch = sys.stdin.read(1)
	finally:
		termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
	return ch

try:
	print("w - Forward | a - Left | s - Backward | d - Right | x - Pause | i - slower | o - faster | q - Quit") 
	while True:
		cmd = get_key().lower()
		if cmd == 'w':
			ser.write(b'w')
			print("[CMD] Forward")
		elif cmd == 'a':
			ser.write(b'a')
			print("[CMD] Left")
		elif cmd == 'd':
			ser.write(b'd')
			print("[CMD] Right")
		elif cmd == 's':
			ser.write(b's')
			print("[CMD] Backward")
		elif cmd == 'i':
			speed = max(0, speed - 10)
			ser.write(b'I')
			print(f"[CMD] speed - 10 = {speed}")
		elif cmd == 'o':
			speed = min(255, speed + 10)
			ser.write(b'O')
			print(f"[CMD] speed + 10 = {speed}")
		elif cmd == 'x':
			ser.write(b'x')
			print("[CMD] stop")
		elif cmd == 'q':
			ser.write(b'x')
			print("[CMD] EXIT")
			break
		else:
			print("Invalid cmd")

except KeyboardInterrupt:
	ser.write(b'q')
	print("\nStopped")
finally:
	ser.close()
