import numpy as np
import cv2
from scipy.interpolate import splprep, splev, splder

SHOW = True


def normalize(contour, column=0):
    normalizer = np.max(contour[0, :, column]) - np.min(contour[0, :, column])
    contour[:, :, 0] = contour[:, :, 0] - np.min(contour[0, :, 0])
    contour[:, :, 1] = contour[:, :, 1] - np.min(contour[0, :, 1])
    return contour / normalizer


def shift(arr, val):
    if val != 0:
        new_arr = np.zeros(arr.shape)
        new_arr[val:, :] = arr[:-val, :]
        new_arr[:val, :] = arr[-val:, :]
        return new_arr
    else:
        return arr


def bbox(contour):
    x, y = contour.T
    width = np.max(x) - np.min(x)
    height = np.max(y) - np.min(y)
    return width, height


def bbox_area(contour):
    rect = bbox(contour)
    return rect[0] * rect[1]


def show_image(img):
    if SHOW:
        cv2.imshow('image', img)
        cv2.waitKey(0)
        cv2.destroyWindow('image')


# turn model screenshot into bitmap sketch
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


# vectorification for model images
def vectorify(image_file, precision=0.0001, smoothing=5000, resolution=50,
              colors=[7, 42, 77, 255], diff_threshold=5, bspline_degree=3,
              n_contours=1):
    image = cv2.imread(image_file)
    # show_image(image)
    # image = cv2.resize(image, (100, 100))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # min_width = min_contour_dim * image.shape[0]
    # min_height = min_contour_dim * image.shape[1]

    images = []
    for c in colors:
        temp = gray.copy()
        color_filter = np.abs(temp - c) > diff_threshold
        temp[color_filter] = 0
        temp[~color_filter] = 255
        if c == 255:
            temp = cv2.bitwise_not(temp)
        images.append(temp)

    all_contours = []
    for image in images:
        image = cv2.medianBlur(image, 31)  # cv2.GaussianBlur(image, (21, 21), 0)
        # show_image(image)
        # find contours in the edged image, keep only the largest
        # ones, and initialize our screen contour
        im2, contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # show_image(im2)
        imcontour = cv2.drawContours(np.ones(image.shape), contours, -1, (0, 0, 0), 3)
        # show_image(imcontour)
        # contours = contours[1:]
        if len(contours) != 0:
            contours_bbox = np.array(list(map(lambda x: bbox(x), contours)))
            # width_filter = contours_bbox[:, 0] > min_width
            # height_filter = contours_bbox[:, 1] > min_height
            # areas = contours_bbox[:, 0] * contours_bbox[:, 1]

            # contours = [c for (c, f1, f2) in zip(contours, width_filter, height_filter) if f1 or f2]
            contours = sorted(contours, key=lambda x: bbox_area(x), reverse=True)
            contours = contours[0:n_contours]

            # linear_contours = []
            # for c in contours:
            #     # approximate the contour
            #     peri = cv2.arcLength(c, True)
            #     approx = cv2.approxPolyDP(c, precision * peri, False)
            #     linear_contours.append(approx)
            moresmooth = []
            for contour in contours:
                x, y = contour.T
                # Convert from numpy arrays to normal arrays
                x = x.tolist()[0]
                y = y.tolist()[0]
                # print(len(x))
                if len(x) > 1:
                    # https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.interpolate.splprep.html
                    # k = min(bspline_degree, len(x))
                    tck, u = splprep([x, y], u=None, s=smoothing, per=0, k=bspline_degree)
                    # https://docs.scipy.org/doc/numpy-1.10.1/reference/generated/numpy.linspace.html
                    u_new = np.linspace(u.min(), u.max(), resolution)
                    # https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.interpolate.splev.html
                    x_new, y_new = splev(u_new, tck, der=0)
                    # xp, yp = splev(u_new, tck, der=1)
                    # xpp, ypp = splev(u_new, tck, der=2)
                    # curvature = np.abs(xp * ypp - yp * xpp) / np.power(xp ** 2 + yp ** 2, 3 / 2)
                    # u_sorted_by_curvature = np.array([u for _, u in
                    #                                   sorted(zip(curvature.tolist(), u_new.tolist()),
                    #                                          key=lambda pair: pair[0], reverse=True)])[
                    #                         :resolution]
                    # print(u_sorted_by_curvature.shape)
                    # u_final = sorted(u_sorted_by_curvature.tolist(), reverse=False)
                    # x_new, y_new = splev(np.array(u_final), tck, der=0)
                    # Convert it back to numpy format for opencv to be able to display it
                    res_array = [[[int(i[0]), int(i[1])]] for i in zip(x_new, y_new)]
                    all_contours.append(np.array(res_array, dtype=np.int32))
                    moresmooth.append(np.array(res_array, dtype=np.int32))
            # im = cv2.drawContours(np.ones(image.shape), np.array(moresmooth, dtype=np.int32), -1, (0, 0, 0), 3)
        else:
            # if no outline for specific part, add complete outline
            all_contours.append(all_contours[0])

    # im = cv2.drawContours(np.ones(image.shape) * 255, np.array(all_contours, dtype=np.int32), -1, (0, 0, 0), 2)  # 2
    # show_image(im)
    # im = im[cropping:im.shape[0] + cropping, cropping:im.shape[1] + cropping]
    # im = cv2.resize(im, output_dim)
    # cv2.imwrite(out_file, im)
    return np.array(all_contours)


# variant of vectorify for rgb images
def process_rgb_sketch(image, precision=0.0001, smoothing=5000, resolution=50, diff_threshold=30, bspline_degree=3,
                       n_contours=1,
                       colors=[[255, 255, 255], [0, 255, 0], [0, 0, 255], [255, 0, 0]]):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    show_image(image)
    images = []
    for c in colors:
        lower = np.array(c) - diff_threshold
        upper = np.array(c) + diff_threshold
        mask = cv2.inRange(image.copy(), lower, upper)
        if c == [255, 255, 255]:
            mask = cv2.bitwise_not(mask)
        show_image(mask)
        images.append(mask)

    all_contours = []
    for image in images:
        # image = cv2.medianBlur(image, 31)  # cv2.GaussianBlur(image, (21, 21), 0)
        show_image(image)
        # find contours in the edged image, keep only the largest
        # ones, and initialize our screen contour
        im2, contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # show_image(im2)
        imcontour = cv2.drawContours(np.ones(image.shape), contours, -1, (0, 0, 0), 3)
        show_image(imcontour)
        # contours = contours[1:]
        if len(contours) != 0:
            contours_bbox = np.array(list(map(lambda x: bbox(x), contours)))
            contours = sorted(contours, key=lambda x: bbox_area(x), reverse=True)
            contours = contours[0:n_contours]
            moresmooth = []
            for contour in contours:
                x, y = contour.T
                # Convert from numpy arrays to normal arrays
                x = x.tolist()[0]
                y = y.tolist()[0]
                # print(len(x))
                if len(x) > 1:
                    tck, u = splprep([x, y], u=None, s=smoothing, per=0, k=bspline_degree)
                    u_new = np.linspace(u.min(), u.max(), resolution)
                    x_new, y_new = splev(u_new, tck, der=0)
                    res_array = [[[int(i[0]), int(i[1])]] for i in zip(x_new, y_new)]
                    all_contours.append(np.array(res_array, dtype=np.int32))
                    moresmooth.append(np.array(res_array, dtype=np.int32))
        else:
            # if no outline for specific part, add complete outline
            all_contours.append(all_contours[0])

    im = cv2.drawContours(np.ones(image.shape) * 255, np.array(all_contours, dtype=np.int32), -1, (0, 0, 0), 2)  # 2
    show_image(im)
    return np.array(all_contours)


# process contours
def process_contours(contours):
    n_contours = contours.shape[0]
    for j in range(n_contours):
        min_y_ind = np.argmin(contours[j, :, 1])
        print(min_y_ind)
        contours[j] = shift(contours[j], min_y_ind)
    contours = normalize(contours)
    return contours


if __name__ == "__main__":
    n_contours = 4
    imagepath = "test_sketch.png"
    image = cv2.imread(imagepath)
    # cv2 uses BGR
    contours = process_rgb_sketch(image, colors=[[255, 255, 255], [0, 255, 48], [236, 0,39], [0, 0, 255]],
                                  precision=0.00001, smoothing=10, resolution=200, diff_threshold=5, bspline_degree=1,
                                  n_contours=1)
    print(contours.shape)
    contours = contours[:, :, 0, :]
    print(process_contours(contours).shape)
