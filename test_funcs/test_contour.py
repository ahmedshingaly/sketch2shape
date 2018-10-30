from skimage import exposure
import numpy as np
import cv2
from scipy.interpolate import splprep, splev
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt

show_steps = False
image = cv2.imread("screenshot.png")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
colors = [7, 42, 77, 255]
images = []
diff_threshold = 5
percentile = 90
bspline_degree = 3
precision = 0.00001
smoothing = 1000
resolution = 10000


def show_image(img):
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyWindow('image')


for c in colors:
    temp = gray.copy()
    color_filter = np.abs(temp - c) > diff_threshold
    temp[color_filter] = 255
    temp[~color_filter] = 0
    images.append(temp)
    if show_steps: show_image(temp)

all_contours = []
for image in images:
    # gray = cv2.bilateralFilter(image, 11, 3, 5)
    if show_steps: show_image(image)
    edged = cv2.Canny(image, 30, 200, 200)
    if show_steps: show_image(edged)
    # find contours in the edged image, keep only the largest
    # ones, and initialize our screen contour
    im2, contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #contours = contours[1:]
    if len(contours) != 0:
        contours1 = sorted(contours, key=cv2.contourArea, reverse=True)
        contours_areas = np.array(list(map(lambda x: cv2.contourArea(x), contours1)))
        area_cutoff = np.percentile(contours_areas, percentile)
        area_filter = contours_areas > area_cutoff
        contours1 = [c for (c, f) in zip(contours1, area_filter) if f]
        contours2 = sorted(contours, key=lambda x: cv2.arcLength(x, False), reverse=True)
        contours_lengths = np.array(list(map(lambda x: cv2.arcLength(x, False), contours)))
        length_cutoff = np.percentile(contours_lengths, percentile)
        length_filter = contours_lengths > length_cutoff
        contours2 = [c for (c, f) in zip(contours2, length_filter) if f]
        contours = []
        contours.extend(contours1)
        contours.extend(contours2)

        im = cv2.drawContours(image.copy(), contours, -1, (150, 150, 150), 3)
        if show_steps: show_image(im)
        smoothened = []
        for c in contours:
            # approximate the contour
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, precision * peri, False)
            smoothened.append(approx)

        im = cv2.drawContours(image.copy(), smoothened, -1, (150, 150, 150), 3)
        if show_steps: show_image(im)

        moresmooth = []
        for contour in smoothened:
            x, y = contour.T
            # Convert from numpy arrays to normal arrays
            x = x.tolist()[0]
            y = y.tolist()[0]
            if len(x) > 1:
                # https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.interpolate.splprep.html
                # k = min(bspline_degree, len(x))
                tck, u = splprep([x, y], u=None, s=smoothing, per=0, k=bspline_degree)
                # https://docs.scipy.org/doc/numpy-1.10.1/reference/generated/numpy.linspace.html
                u_new = np.linspace(u.min(), u.max(), resolution)
                # https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.interpolate.splev.html
                x_new, y_new = splev(u_new, tck, der=0)
                # Convert it back to numpy format for opencv to be able to display it
                res_array = [[[int(i[0]), int(i[1])]] for i in zip(x_new, y_new)]
                all_contours.append(np.array(res_array, dtype=np.int32))
                moresmooth.append(np.array(res_array, dtype=np.int32))
        im = cv2.drawContours(np.ones(image.shape), np.array(moresmooth, dtype=np.int32), -1, (0, 0, 0), 3)
        if show_steps: show_image(im)

all_contours = sorted(all_contours, key=lambda x: cv2.arcLength(x, False), reverse=True)
im = cv2.drawContours(np.ones(image.shape), np.array(all_contours, dtype=np.int32), -1, (0, 0, 0), 2)
cropping = 20
im = im[cropping:im.shape[0]+cropping,cropping:im.shape[1]+cropping]
cv2.imshow("Sketch", im)
cv2.waitKey(0)
cv2.destroyWindow("Sketch")
