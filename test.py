import os
import cv2
import math
import numpy as np

import imutils

from scipy.spatial import distance as dist


def calculate_angle_and_length_between_two_points(x1, y1, x2, y2):
    length = round(math.sqrt((x2-x1)**2 + (y2-y1)**2), 1)
    angle = round(math.degrees(math.atan2(y2-y1, x2-x1)), 1)
    return length, angle


def resize_to_desired_max_size_for_processing(img, max_size=1200):

    og_height, og_width = img.shape[:2]

    inter_ploation = (
        cv2.INTER_CUBIC
        if og_height < max_size and og_width < max_size
        else cv2.INTER_AREA
    )
    img_s_resized = (
        (int(og_width * max_size / og_height), max_size)
        if og_width < og_height
        else (max_size, int(og_height * max_size / og_width))
    )

    img = cv2.resize(img, img_s_resized, interpolation=inter_ploation)
    return img


def _convert_in_binary_image(gray_image, bf=7, bf_v=120, bs=19, c=-5, sigma=0.33):

    bf_image = cv2.bilateralFilter(gray_image, bf, bf_v, bf_v)

    img_threshold = cv2.adaptiveThreshold(
        cv2.bitwise_not(bf_image),
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        bs,
        c,
    )

    intensity = np.median(bf_image)
    lower_threshold = int(max(0, (1.0 - sigma) * intensity))
    upper_threshold = int(min(255, (1.0 + sigma) * intensity))

    img = cv2.bitwise_and(
        cv2.bitwise_not(cv2.Canny(bf_image, lower_threshold, upper_threshold)),
        img_threshold,
    )

    return img


def get_perpsective_pts_from_grid_table_lines_in_image(img, bn_img):

    height, width = bn_img.shape[:2]

    minLineLength = int(width*0.40)
    lines = cv2.HoughLinesP(
        image=bn_img,
        rho=1,
        theta=np.pi/180, 
        threshold=int(minLineLength*0.60),  
        minLineLength=minLineLength,
        maxLineGap=int(minLineLength*0.05)
    )

    lines_img = np.zeros_like(bn_img)
    h_lines, v_lines = [], []
    num_lines, _, _ = lines.shape
    for i in range(num_lines):

        line_info = lines[i][0]
        x1, y1 = line_info[0], line_info[1]
        x2, y2 = line_info[2], line_info[3]
        
        length, angle = calculate_angle_and_length_between_two_points(
            x1, y1, x2, y2
        )
        line_info = [(x1, y1), (x2, y2), length, angle]
        if abs(angle) > 45:
            v_lines.append(line_info)
            cv2.line(lines_img, (x1, y1), (x2, y2), 255, 2, cv2.LINE_AA)
        else:
            h_lines.append(line_info)
            cv2.line(lines_img, (x1, y1), (x2, y2), 255, 2, cv2.LINE_AA)
    
    if not v_lines:
        return False, -1

    ## find contours
    contours, _ = cv2.findContours(
        lines_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )

    cnts_img = np.zeros_like(bn_img)
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.1*cv2.arcLength(cnt, True), True)
        x, y, w, h = cv2.boundingRect(cnt)
        sq_ratio_err = abs(1 - w/h)
        if len(approx) == 4 and sq_ratio_err < 0.50 and w < width//2 and h < height // 2:
            cv2.drawContours(cnts_img, [cnt], -1, 255, 2)

    cnts_img = cv2.dilate(cnts_img, np.ones((30, 30), dtype=np.uint8), 1)
    cnts_img = cv2.erode(cnts_img, np.ones((10, 10), dtype=np.uint8), 1)

    contours, _ = cv2.findContours(
        cnts_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not len(contours):
        return False, -1

    contours.sort(key=cv2.contourArea, reverse=True)

    cnt = contours[0]
    approx = cv2.approxPolyDP(cnt, 0.1*cv2.arcLength(cnt, True), True)

    return True, approx.reshape(4, 2)


def order_points(pts):

    xSorted = pts[np.argsort(pts[:, 0]), :]
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]

    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    (tl, bl) = leftMost
    
    D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
    (br, tr) = rightMost[np.argsort(D)[::-1], :]

    return np.array([tl, tr, br, bl], dtype = "float32")
    

def four_point_transform(image, pts):

    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)

    # new_image = get_resized_image_to_fit_in_transformation_quad(image, rect)
    perpective_corrected_img = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return perpective_corrected_img

    
def auto_correct_perspective_of_diamond_paper(image):

    im_h, im_w = image.shape[:2]
    min_area_req = im_h * im_w * 0.50

    # # convert the image to grayscale and blur it slightly
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    bn_img = _convert_in_binary_image(gray, bf=7, bf_v=75, bs=17, c=-3)
    bn_img = cv2.dilate(
        bn_img, np.ones((3, 1), dtype=np.uint8), iterations=1
    )
    bn_img = cv2.dilate(
        bn_img, np.ones((1, 3), dtype=np.uint8), iterations=1
    )
    cv2.imshow("intermediate", bn_img)
    cv2.waitKey(0)

    suc, perspective_points = get_perpsective_pts_from_grid_table_lines_in_image(image, bn_img)
    if not suc:
        return image

    ps_img = four_point_transform(image, perspective_points)
    return ps_img
    

def get_square_bboxes_avg_size_in_px(image):

    # # convert the image to grayscale and blur it slightly
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    bn_img = _convert_in_binary_image(gray, bf=7, bf_v=75, bs=19, c=-3)
    bn_img = cv2.dilate(
        bn_img, np.ones((5, 1), dtype=np.uint8), iterations=1
    )
    bn_img = cv2.dilate(
        bn_img, np.ones((1, 5), dtype=np.uint8), iterations=1
    )
    cv2.imshow("intermediate", bn_img)
    cv2.waitKey(0)

    height, width = bn_img.shape[:2]

    minLineLength = int(width*0.50)
    lines = cv2.HoughLinesP(
        image=bn_img,
        rho=1,
        theta=np.pi/180, 
        threshold=int(minLineLength*0.60),  
        minLineLength=minLineLength,
        maxLineGap=int(minLineLength*0.05)
    )

    lines_img = np.zeros_like(bn_img)
    num_lines, _, _ = lines.shape
    
    for i in range(num_lines):

        line_info = lines[i][0]
        x1, y1 = line_info[0], line_info[1]
        x2, y2 = line_info[2], line_info[3]
        
        cv2.line(lines_img, (x1, y1), (x2, y2), 255, 2, cv2.LINE_AA)

    ## find contours
    contours, _ = cv2.findContours(
        lines_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )

    sq_size_w, sq_size_h = [], []

    cnts_img = np.zeros_like(bn_img)
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.1*cv2.arcLength(cnt, True), True)
        x, y, w, h = cv2.boundingRect(cnt)
        sq_ratio_err = abs(1 - w/h)
        if len(approx) == 4 and sq_ratio_err < 0.15 and w < width//2 and h < height // 2:
            sq_size_w.append(w)
            sq_size_h.append(h)
            cv2.drawContours(cnts_img, [cnt], -1, 255, 2)

    if not (sq_size_w and sq_size_h):
        return -1, -1

    avg_w = np.quantile(sq_size_w, 0.5)
    avg_h = np.quantile(sq_size_h, 0.5)
    
    cv2.imshow("intermediate", cnts_img)
    cv2.waitKey(0)

    return avg_w, avg_h


if __name__ == "__main__":

    img_folder = "images"
    all_img_files = os.listdir(img_folder) 

    cv2.namedWindow("input", cv2.WINDOW_NORMAL)
    # cv2.namedWindow("output", cv2.WINDOW_NORMAL)
    cv2.namedWindow("intermediate", cv2.WINDOW_NORMAL)

    for img_f in all_img_files:

        og_img = cv2.imread(f"{img_folder}/{img_f}")
        og_img = resize_to_desired_max_size_for_processing(og_img, max_size=1408)
        cv2.imshow("input", og_img)

        perspective_img = auto_correct_perspective_of_diamond_paper(og_img)
        # cv2.imwrite(f"output/{img_f}", perspective_img)
        perspective_img = resize_to_desired_max_size_for_processing(perspective_img, max_size=1408)

        cv2.imshow("intermediate", perspective_img)
        k = cv2.waitKey(0)
        if k == ord("q"):
            break


        ## get sq box size now
        avg_w, avg_h = get_square_bboxes_avg_size_in_px(perspective_img)
        print(avg_w, avg_h)



"""
def test():
    ## pair all lines having similar length and apprx same angle, keep the set with max entries

    # sub_v_pairs = []

    # x_pos = set()
    # for line in v_lines:
    #     x_pos.add(line[0][0])
    #     # print(line[2], line[3], "X POS =>", line[0][0])
    # x_pos = sorted(x_pos)

    # print("\n", x_pos)

    # ## merged by sub seq diff of 5
    # new_x_pos = [x_pos[0]]
    # for xp in x_pos[1:]:
    #     if xp - new_x_pos[-1] < 15:
    #         new_x_pos[-1] = xp
    #     else:
    #         new_x_pos.append(xp)

    # num_lines = len(new_x_pos)

    # if num_lines < 4:
    #     print("***************")
    #     return []
    
    # diff_median = int(
    #     np.quantile([new_x_pos[i] - new_x_pos[i-1]  for i in range(1, num_lines)], 0.5)
    # )

    # print([new_x_pos[i] - new_x_pos[i-1]  for i in range(1, num_lines)], diff_median)

    # # dilate and erode
    # kernel = np.ones((3, diff_median), dtype=np.uint8)
    # lines_img = cv2.dilate(lines_img, kernel, iterations=1)
    # lines_img = cv2.erode(lines_img, kernel, iterations=1)

    # cv2.imshow("intermediate", lines_img)

"""
