import serial
import time

ser = serial.Serial('/dev/ttyACM0', 9600, timeout=1)
time.sleep(2)

def send(cmd):
	ser.write(cmd.encode())
	print(f"Sent: {cmd}")
	time.sleep(0.1)

send('L'); time.sleep(2)
send('S'); time.sleep(1)

send('R'); time.sleep(2)
send('S'); time.sleep(1)

send('L'); time.sleep(2)
send('S'); time.sleep(1)

send('R'); time.sleep(2)
send('S')

ser.close()
print("Done!")
