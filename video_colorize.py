import sys, os
import random
import cv2
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from colorize import colorize

def nearby(x, y, n, m, d = 1):
	neighbour = []
	for i in range(max(0, x - d), min(n, x + d + 1)):
		for j in range(max(0, y - d), min(m, y + d + 1)):
			if (i != x) or (j != y):
				neighbour.append([i, j])
	return neighbour

def Lucas_Kanade(last_gray, gray, last_output):
	n, m = gray.shape[0], gray.shape[1]
	size = n * m
	sketch = gray.copy()
	grad_x, grad_y = np.gradient(last_gray[:, :, 0])
	labeled = {}
	for sampling in range(size // 8):
		x, y = random.randint(0, n - 1), random.randint(0, m - 1)
		while (x, y) in labeled:
			x, y = random.randint(0, n - 1), random.randint(0, m - 1)
		labeled[x, y] = True
		neighbour = nearby(x, y, n, m)
		A = []
		b = []
		for pixel in neighbour:
			A.append([grad_x[pixel[0], pixel[1]], grad_y[pixel[0], pixel[1]]])
			b.append([last_gray[pixel[0], pixel[1]] - last_gray[pixel[0], pixel[1]]])
		try:
			v = scipy.linalg.solve(np.dot(np.transpose(A), A), np.dot(np.transpose(A), b))
		except:
			v = [0, 0]
		if v[0] == None:
			v[0] = 0
		if v[1] == None:
			v[1] = 0
		print(v[0], v[1])
		x_new, y_new = int(x + v[0]), int(y + v[1])
		if x_new < 0:
			x_new = 0
		if x_new >= n:
			x_new = n - 1
		if y_new < 0:
			y_new = 0
		if y_new >= m:
			y_new = m - 1
		sketch[x_new, y_new, 1] = output[x, y, 1]
		sketch[x_new, y_new, 2] = output[x, y, 2]
	return sketch

if __name__ == "__main__":
	try:
		frame_num = int(sys.argv[1])
	except:
		print("Usage: video_colorize <frame_num>")
		exit(0)
	
	gray, sketch, output = None, None, None
	if not os.path.exists("colored"):
		os.mkdir("colored")
	for id in range(frame_num):
		last_gray = gray
		try:
			gray = cv2.imread(os.path.join(".", "frame", "frame%d.bmp" % id))
			gray = cv2.cvtColor(gray, cv2.COLOR_BGR2YUV)
			gray = gray / 255.0
		except:
			print("Failed to read images.")
			exit(0)
		
		if not os.path.exists(os.path.join(".", "sketch", "sketch%d.bmp" % id)):
			sketch = Lucas_Kanade(last_gray, gray, output)
		else:
			try:
				sketch = cv2.imread(os.path.join(".", "sketch", "sketch%d.bmp" % id))
				sketch = cv2.cvtColor(sketch, cv2.COLOR_BGR2YUV)
				sketch = sketch / 255.0
			except:
				print("Failed to read images.")
				exit(0)
		assert gray.shape == sketch.shape, "The two images should share the same size."
		
		output = colorize(gray, sketch)
		output_image = cv2.cvtColor((np.clip(output, 0., 1.) * 255).astype(np.uint8), cv2.COLOR_YUV2BGR)
		cv2.imwrite(os.path.join(".", "colored", "colored%d.bmp" % id), output_image)
