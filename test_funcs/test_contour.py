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

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.bilateralFilter(gray, 11, 100,100)
print(gray)
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
cv2.imshow("Contours", im2)
cv2.waitKey(0)
cv2.destroyWindow("Contours")
contours = sorted(contours, key=lambda x: cv2.arcLength(x, False), reverse=True)
contours = contours[:10]
print(contours[0].shape)
contours = sorted(contours, key=lambda x: x.shape[0], reverse=True)
contours = contours[:]
j = 0
for c in contours:
    j += 1
    im = cv2.drawContours(image.copy(), [c], -1, (0, 255, 0), 3)
    cv2.imshow(str(j), im)
    cv2.waitKey(0)
    cv2.destroyWindow(str(j))
# hierarchy = sorted(hierarchy, key=lambda x: x[1], reverse=True)
# contours = [contours[i] for i in hierarchy[0][:,0]]

# screenCnt = []
# for c in contours:
#     segments = []
#     print(c.shape)
#     for i in range(len(c) - 1):
#         segments.append([c[i], c[i + 1]])
#     screenCnt.append(np.array(segments))
# print(screenCnt)
# contours = sorted(screenCnt, key=lambda x: cv2.arcLength(x, False), reverse=True)

im = cv2.drawContours(image.copy(), contours[:20], -1, (0, 255, 0), 3)
cv2.imshow("10 Longest Contours", im)
cv2.waitKey(0)
cv2.destroyWindow("10 Longest Contours")
smoothened = []
for c in contours[:5]:
    # approximate the contour
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.0001 * peri, False)
    smoothened.append(approx)

moresmooth = []
for contour in smoothened[:5]:
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
    moresmooth.append(np.asarray(res_array, dtype=np.int32))

im = cv2.drawContours(np.ones(image.shape), moresmooth, -1, (0, 0, 0), 3)
cv2.imshow("Sketch", im)
cv2.waitKey(0)
cv2.destroyWindow("Sketch")