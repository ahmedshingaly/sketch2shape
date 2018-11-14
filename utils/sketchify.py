import numpy as np
import cv2
from scipy.interpolate import splprep, splev


def bbox(contour):
    x, y = contour.T
    width = np.max(x) - np.min(x)
    height = np.max(y) - np.min(y)
    return width, height


def sketchify(image_file, out_file, min_contour_dim=0.3, precision=0.0001, smoothing=5000, resolution=1000,
              output_dim=(200, 200), colors=[7, 42, 77, 255], diff_threshold=5, cropping=20, bspline_degree=3):
    image = cv2.imread(image_file)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    min_width = min_contour_dim * image.shape[0]
    min_height = min_contour_dim * image.shape[1]

    images = []
    for c in colors:
        temp = gray.copy()
        color_filter = np.abs(temp - c) > diff_threshold
        temp[color_filter] = 255
        temp[~color_filter] = 0
        images.append(temp)

    all_contours = []
    for image in images:
        # find contours in the edged image, keep only the largest
        # ones, and initialize our screen contour
        im2, contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[1:]
        if len(contours) != 0:
            contours_bbox = np.array(list(map(lambda x: bbox(x), contours)))
            width_filter = contours_bbox[:, 0] > min_width
            height_filter = contours_bbox[:, 1] > min_height
            contours = [c for (c, f1, f2) in zip(contours, width_filter, height_filter) if f1 or f2]

            linear_contours = []
            for c in contours:
                # approximate the contour
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, precision * peri, False)
                linear_contours.append(approx)
            moresmooth = []
            for contour in linear_contours:
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
    im = cv2.drawContours(np.ones(image.shape) * 255, np.array(all_contours, dtype=np.int32), -1, (0, 0, 0), 2)
    im = im[cropping:im.shape[0] + cropping, cropping:im.shape[1] + cropping]
    im = cv2.resize(im, output_dim)
    cv2.imwrite(out_file, im)
