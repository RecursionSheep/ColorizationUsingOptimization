import cv2
import sys, os
import numpy as np

if __name__ == "__main__":
	try:
		video_dir = sys.argv[1]
	except:
		print("Usage: video_cut <video>")
		exit(0)
	
	video = cv2.VideoCapture(video_dir)
	cnt = 0
	if not os.path.exists("frame"):
		os.mkdir("frame")
	while True:
		succ, frame = video.read()
		if not succ:
			break
		cnt += 1
		cv2.imwrite(os.path.join(".", "frame", "frame%d.bmp" % cnt), frame)
	
	video.release()
