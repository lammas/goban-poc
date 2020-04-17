import sys
import cv2
import numpy as np
import scipy
import scipy.spatial.distance

WINDOW_NAME = 'demo'


def debug(img):
	cv2.namedWindow('debug', cv2.WINDOW_NORMAL)
	cv2.resizeWindow('debug', 1800, 1200)
	cv2.imshow('debug', img)


def preprocess(img):
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	gray = cv2.medianBlur(gray, 7)
	thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
	return thresh


def findCorners(output, original, thresh):
	gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
	gray_fp = np.float32(gray)
	dst = cv2.cornerHarris(gray_fp, 16, 27, 0.22)
	dst = cv2.dilate(dst, None)
	ret, dst = cv2.threshold(dst, 0.01 * dst.max(), 255, 0)
	dst = np.uint8(dst)
	ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
	corners = cv2.cornerSubPix(gray, np.float32(centroids), (50, 50), (-1, -1), criteria)
	for corner in corners:
		cv2.drawMarker(output, (corner[0], corner[1]), (0, 0, 255), markerType=cv2.MARKER_STAR, markerSize=16, thickness=2, line_type=cv2.LINE_AA)
	return np.round(corners).astype('int')


def findCircles(output, original, thresh):
	circles = cv2.HoughCircles(thresh, cv2.HOUGH_GRADIENT, 1, 10, param1=100, param2=22, minRadius=20, maxRadius=50)
	if circles is not None:
		circles = np.round(circles[0, :]).astype('int')
		for (x, y, r) in circles:
			cv2.circle(output, (x, y), r, (0, 255, 0), 4)
			cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
	return circles


def dist2(p1, p2):
	return (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2


def merge(points, d):
	ret = []
	d2 = d * d
	n = len(points)
	taken = [False] * n
	for i in range(n):
		if not taken[i]:
			count = 1
			point = [points[i][0], points[i][1]]
			taken[i] = True
			for j in range(i + 1, n):
				if dist2(points[i], points[j]) < d2:
					point[0] += points[j][0]
					point[1] += points[j][1]
					count += 1
					taken[j] = True
			point[0] /= count
			point[1] /= count
			ret.append((point[0], point[1]))
	return np.array(ret)


def findGrid(axis):
	steps = dict()
	axis = sorted(list(set(axis)))
	print(repr(axis))
	for i in range(1, len(axis)):
		step = round(axis[i] - axis[i - 1], -1)
		if step not in steps:
			steps[step] = 0
		steps[step] += 1
	print(repr(steps))


def main():
	cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
	cv2.resizeWindow(WINDOW_NAME, 1800, 1200)

	# cv2.namedWindow('thresh', cv2.WINDOW_NORMAL)
	# cv2.resizeWindow('thresh', 1800, 1200)

	# img = cv2.imread('data/board.jpg')
	# img = cv2.imread('data/board_stones1.jpg')
	img = cv2.imread('data/board_stones4.jpg')
	output = img

	thresh = preprocess(img)

	output = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
	# cv2.imshow('thresh', thresh)

	corners = findCorners(output, img, thresh)
	circles = findCircles(output, img, thresh)

	points = np.zeros(shape=(len(circles) + len(corners), 2), dtype='int')
	i = 0
	for (x, y) in corners:
		points[i] = [x, y]
		i += 1
	for (x, y, r) in circles:
		points[i] = [x, y]
		i += 1

	output = img
	simplified = merge(points, 20)
	simplified = np.round(simplified).astype('int')
	for (x, y) in simplified:
		cv2.drawMarker(output, (int(x), int(y)), (255, 255, 0), markerType=cv2.MARKER_SQUARE, markerSize=16, thickness=2, line_type=cv2.LINE_AA)

	cv2.imshow('demo', output)
	cv2.waitKey()
	cv2.destroyAllWindows()
	return 0


if __name__ == '__main__':
	sys.exit(main())
