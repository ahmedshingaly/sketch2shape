from skimage import exposure
import numpy as np
import cv2
from scipy.interpolate import splprep, splev
from sklearn.cluster import KMeans

image = cv2.imread("screenshot.png")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
print(gray.shape)
# ncolors = 4 # including white
# kmeans= KMeans(ncolors)
# kmeans.fit(np.reshape(gray,(np.size(gray),1)))
# print(kmeans.cluster_centers_)
# get planes
colors = [7, 42, 77, 255]
images = []
diff_threshold = 5
for c in colors:
    temp = gray.copy()
    color_filter = np.abs(temp - c) > diff_threshold
    temp[color_filter] = 255
    temp[~color_filter] = 0
    images.append(temp)
    cv2.imshow(str(c), temp)
    cv2.waitKey(0)
    cv2.destroyWindow(str(c))

all_contours = []
for image in images:
    gray = cv2.bilateralFilter(image, 11, 100, 100)
    cv2.imshow('image', gray)
    cv2.waitKey(0)
    cv2.destroyWindow('image')
    edged = cv2.Canny(gray, 150, 250, 200)
    cv2.imshow('Edged', edged)
    cv2.waitKey(0)
    cv2.destroyWindow('Edged')
    # find contours in the edged image, keep only the largest
    # ones, and initialize our screen contour
    im2, contours, hierarchy = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[:]
    # cv2.imshow("Contours", im2)
    # cv2.waitKey(0)
    # cv2.destroyWindow("Contours")
    # contours = sorted(contours, key=lambda x: cv2.arcLength(x, False), reverse=True)
    contours1 = sorted(contours, key=cv2.contourArea, reverse=True)[:2]
    contours2 = sorted(contours, key=lambda x: cv2.arcLength(x, False), reverse=True)[:2]
    contours = []
    contours.extend(contours1)
    contours.extend(contours2)
    im = cv2.drawContours(image.copy(), contours1, -1, (150, 150, 150), 3)
    im = cv2.drawContours(im.copy(), contours2, -1, (150, 150, 150), 3)
    cv2.imshow("10 Longest Contours", im)
    cv2.waitKey(0)
    cv2.destroyWindow("10 Longest Contours")
    smoothened = []
    for c in contours:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.0001 * peri, False)
        smoothened.append(approx)
    im = cv2.drawContours(image.copy(), smoothened, -1, (150, 150, 150), 3)
    cv2.imshow("Hey", im)
    cv2.waitKey(0)
    cv2.destroyWindow("Hey")
    moresmooth = []
    for contour in smoothened:
        x, y = contour.T
        # Convert from numpy arrays to normal arrays
        x = x.tolist()[0]
        y = y.tolist()[0]
        # https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.interpolate.splprep.html
        tck, u = splprep([x, y], u=None, s=10000, per=0, k=3)
        # https://docs.scipy.org/doc/numpy-1.10.1/reference/generated/numpy.linspace.html
        u_new = np.linspace(u.min(), u.max(), 10000)
        # https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.interpolate.splev.html
        x_new, y_new = splev(u_new, tck, der=0)
        # Convert it back to numpy format for opencv to be able to display it
        res_array = [[[int(i[0]), int(i[1])]] for i in zip(x_new, y_new)]
        all_contours.append(np.array(res_array, dtype=np.int32))
        moresmooth.append(np.array(res_array, dtype=np.int32))
    im = cv2.drawContours(np.ones(image.shape), np.array(moresmooth, dtype=np.int32), -1, (0, 0, 0), 3)
    cv2.imshow("Sketch", im)
    cv2.waitKey(0)
    cv2.destroyWindow("Sketch")
    # all_contours.append(moresmooth)
print(len(all_contours))
all_contours = sorted(all_contours, key=lambda x: cv2.arcLength(x, False), reverse=True)
im = cv2.drawContours(np.ones(image.shape), np.array(all_contours, dtype=np.int32), -1, (0, 0, 0), 3)
cv2.imshow("Sketch", im)
cv2.waitKey(0)
cv2.destroyWindow("Sketch")
