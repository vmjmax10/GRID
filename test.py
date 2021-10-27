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


def get_perpsective_from_grid_table_lines_in_image(img, bn_img):

    h, w = bn_img.shape[:2]

    minLineLength = int(w*0.40)
    lines = cv2.HoughLinesP(
        image=bn_img,
        rho=1,
        theta=np.pi/180, 
        threshold=int(minLineLength*0.60),  
        minLineLength=minLineLength,
        maxLineGap=20
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
            # cv2.line(lines_img, (x1, y1), (x2, y2), 255, 2, cv2.LINE_AA)
    
    x_pos = set()
    for line in v_lines:
        x_pos.add(line[0][0])
        # print(line[2], line[3], "X POS =>", line[0][0])
    x_pos = sorted(x_pos)

    ## merged by sub seq diff of 5

    new_x_pos = []
    for xp in x_pos:

        pass

    
    print(x_pos)

    ## dilate and erode
    # kernel = np.ones((3, 50), dtype=np.uint8)
    # lines_img = cv2.dilate(lines_img, kernel, iterations=1)
    # lines_img = cv2.erode(lines_img, kernel, iterations=1)



    cv2.imshow("intermediate", lines_img)

    return lines_img


def auto_correct_perspective_of_diamond_paper(image):

    im_h, im_w = image.shape[:2]
    min_area_req = im_h * im_w * 0.50

    # # convert the image to grayscale and blur it slightly
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    edge_map = _convert_in_binary_image(gray, bf=7, bf_v=75, bs=17, c=-3)
    edge_map = cv2.dilate(
        edge_map, np.ones((3, 1), dtype=np.uint8), iterations=1
    )
    edge_map = cv2.dilate(
        edge_map, np.ones((1, 3), dtype=np.uint8), iterations=1
    )
    cv2.imshow("intermediate", edge_map)
    cv2.waitKey(0)

    perspective_points = get_perpsective_from_grid_table_lines_in_image(image, edge_map)


def auto_correct_perpective_of_the_paper(image, max_quad_angle=15, min_quad_area_thresh=0.25):

    def angle_between_vectors_degrees(u, v):
        return np.degrees(math.acos(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))))

    def get_angle(p1, p2, p3):

        a = np.radians(np.array(p1))
        b = np.radians(np.array(p2))
        c = np.radians(np.array(p3))

        avec = a - b
        cvec = c - b

        return angle_between_vectors_degrees(avec, cvec)

    def angle_range(quad):
       
        try:
            tl, tr, br, bl = quad
            ura = get_angle(tl[0], tr[0], br[0])
            ula = get_angle(bl[0], tl[0], tr[0])
            lra = get_angle(tr[0], br[0], bl[0])
            lla = get_angle(br[0], bl[0], tl[0])

            angles = [ura, ula, lra, lla]
            final_angle = np.ptp(angles)
            # print(final_angle)

            return final_angle
        except Exception as e:
    
            return 9999         

    def order_points(pts):

        xSorted = pts[np.argsort(pts[:, 0]), :]
        leftMost = xSorted[:2, :]
        rightMost = xSorted[2:, :]

        leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
        (tl, bl) = leftMost
        
        D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
        (br, tr) = rightMost[np.argsort(D)[::-1], :]

        return np.array([tl, tr, br, bl], dtype = "float32")

    def _convert_in_binary_image(gray_image, bf=7, bf_v=120, bs=19, c=-5, sigma=0.33):

        bf_image = cv2.bilateralFilter(gray_image, bf, bf_v, bf_v)

        # cv2.imshow("intermediate", bf_image)
        # cv2.waitKey(0)

        img_threshold = cv2.adaptiveThreshold(
            cv2.bitwise_not(bf_image),
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            bs,
            c,
        )

        # cv2.imshow("intermediate", img_threshold)
        # cv2.waitKey(0)

        intensity = np.median(bf_image)
        lower_threshold = int(max(0, (1.0 - sigma) * intensity))
        upper_threshold = int(min(255, (1.0 + sigma) * intensity))

        # cv2.imshow("intermediate", cv2.bitwise_not(cv2.Canny(bf_image, lower_threshold, upper_threshold)))
        # cv2.waitKey(0)

        img = cv2.bitwise_and(
            cv2.bitwise_not(cv2.Canny(bf_image, lower_threshold, upper_threshold)),
            img_threshold,
        )

        return img

    def _get_stripped_corner_points(
        bn_im, max_keep=25, padding=15, cv_b_rect=cv2.boundingRect
    ):

        h, w = bn_im.shape[:2]

        closing = cv2.morphologyEx(
            bn_im,
            cv2.MORPH_CLOSE,
            np.ones((max(25, h // 20), max(25, w // 20)), dtype=np.uint8),
        )

        opening = cv2.morphologyEx(
            closing,
            cv2.MORPH_OPEN,
            np.ones((max(10, h // 40), max(10, w // 40)), dtype=np.uint8),
        )

        contours, _ = cv2.findContours(
            opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        left, right, top, bottom = 0, w, 0, h

        if contours:

            contours.sort(key=cv2.contourArea, reverse=True)
            xc, yc, wc, hc = cv_b_rect(contours[0])
            left, right, top, bottom = xc, xc + wc, yc, yc + hc

            for cnt in contours[1:max_keep]:

                xc, yc, wc, hc = cv_b_rect(cnt)

                if (
                    (hc > 40 and wc > 50)
                    or abs(wc - (right - left)) < w / 2 
                    or abs(hc - (bottom - top)) < h / 2
                ):

                    if xc < left:
                        left = xc
                    if xc + wc > right:
                        right = xc + wc
                    if yc < top:
                        top = yc
                    if yc + hc > bottom:
                        bottom = yc + hc

        return (
            max(0, left - padding),
            min(w, right + padding),
            max(0, top - padding),
            min(h, bottom + padding),
        )

    def get_document_contour(image):
        
        im_h, im_w = image.shape[:2]
        min_area_req = im_h * im_w * min_quad_area_thresh

        # # convert the image to grayscale and blur it slightly
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        # FLD(gray)

        edge_map = _convert_in_binary_image(gray, bf=7, bf_v=75, bs=17, c=-3)
        edge_map = cv2.dilate(
            edge_map, np.ones((3, 1), dtype=np.uint8), iterations=1
        )
        edge_map = cv2.dilate(
            edge_map, np.ones((1, 3), dtype=np.uint8), iterations=1
        )
        # cv2.imshow("intermediate", edge_map)
        # cv2.waitKey(0)

        get_lines(image, edge_map)





        # cnts, _ = cv2.findContours(edge_map.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        # print("\nstage 1", len(cnts))
        # cnts = [
        #     [cnt, cv2.approxPolyDP(cnt, 0.01*cv2.arcLength(cnt, True), True)] 
        #     for cnt in cnts
        #     if cv2.contourArea(cnt) > 50
        # ]

        # print("stage 2", len(cnts))

        # cnts = sorted(
        #     [[cnt[0], cv2.boundingRect(cnt[0]), cnt[1]] for cnt in cnts if len(cnt[1]) == 4], 
        #     key=lambda x:x[1][2]*x[1][3], 
        #     reverse=True
        # )

        # print("stage 3", len(cnts))

        # disp_img = image.copy()
        # for _, bbox, _ in cnts:
        #     x, y, w, h = bbox
        #     cv2.rectangle(disp_img, (x, y), (x+w, y+h), (0, 0 , 255), thickness=2)

        # cv2.imshow("intermediate", disp_img)
        # cv2.waitKey(0)


        # approx_contours = []
        # tl, tr, bl, br = -1, -1, -1, -1

        # # loop over the contours
        # for cnt, bbox, approx in cnts:
            
        #     x, y, w, h = bbox
        #     sq_ratio =  abs(1 - h/w) 

        #     # print(x, y, w, h, angle_range(approx), sq_ratio)

        #     if sq_ratio < 0.15 and angle_range(approx) < max_quad_angle:
        #         # if approx_contours:
        #         #     if approx_contours[-1][1] < sq_ratio:
        #         #         approx_contours.append([approx, sq_ratio])
        #         #         if len(approx_contours) == 2:
        #         #             break
        #         # else:

        #             # if min_x1 >
                
        #         if w * h > min_area_req:
        #             approx_contours.append([approx, sq_ratio])
        #             break
        #         else:
                    
        #             if isinstance(tl, (tuple, list)):

        #                 print([[a[0][0], a[0][1]] for a in approx])
                        
        #                 t_l, t_r, b_r, b_l = [[a[0][0], a[0][1]] for a in approx]
                        
        #                 if t_l[0] < tl[0] or t_l[1] < tl[1]:
        #                     tl = t_l 

        #                 if t_r[0] > tr[0] or t_r[1] < tr[1]:
        #                     tr = t_r

        #                 if b_r[0] > br[0] or b_r[1] > br[1]:
        #                     br = b_r
                        
        #                 if b_l[0] < bl[0] or b_l[1] > bl[1]:
        #                     bl = b_l

        #             else:
        #                 print([[a[0][0], a[0][1]] for a in approx])
        #                 tl, tr, br, bl = [[a[0][0], a[0][1]] for a in approx]


        # # If we did not find any valid contours, just use the whole image
        # if not approx_contours:

        #     print("NOT FOUND", tl)

        #     if isinstance(tl, (tuple, list)):

        #         print("NOT FgsjksOUND") 

        #         TOP_RIGHT = tuple(tr)
        #         BOTTOM_RIGHT = tuple(br)
        #         BOTTOM_LEFT = tuple(bl)
        #         TOP_LEFT = tuple(tl)

        #     else:
               
        #         TOP_RIGHT = (im_w, 0)
        #         BOTTOM_RIGHT = (im_w, im_h)
        #         BOTTOM_LEFT = (0, im_h)
        #         TOP_LEFT = (0, 0)
            
        #     screenCnt = np.array([[TOP_RIGHT], [BOTTOM_RIGHT], [BOTTOM_LEFT], [TOP_LEFT]])
        #     print(screenCnt)

        # else:
        #     # screenCnt = max(approx_contours, key=cv2.contourArea)
        
        #     screenCnt = approx_contours[-1][0]
        #     print(screenCnt)
            

        # return screenCnt.reshape(4, 2)

    def get_resized_image_to_fit_in_transformation_quad(image, rect):
        
        (tl, tr, br, bl) = rect
        min_x = np.min([int(tl[0]), int(bl[0])])
        max_x = np.min([int(tr[0]), int(br[0])])

        min_y = np.min([int(tl[1]), int(tr[1])])
        max_y = np.min([int(bl[1]), int(br[1])])

        new_image = np.zeros_like(image)
        og_h, og_w = image.shape[:2]

        w_x_margin = (max_x-min_x)*0.20
        new_img_xl = int(max(0, min_x - w_x_margin))
        new_img_xr = int(min(og_w, max_x + w_x_margin))
        new_img_yt = int(max(0, min_y - (max_y-min_y)*0.25))
        new_img_yb = int(min(og_h, max_y + (max_y-min_y)*0.45))

        img_section = image[new_img_yt:new_img_yb, new_img_xl:new_img_xr].copy()
        og_sh, og_sw = img_section.shape[:2]
        
        new_section_image = np.zeros_like(img_section)
        img_resized = cv2.resize(img_section, None, fx=0.60, fy=0.60)
        new_h, new_w = img_resized.shape[:2]

        pad_h, pad_w = (og_sh-new_h)//2, (og_sw-new_w)//2 
        new_section_image[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = img_resized
        
        new_image[new_img_yt:new_img_yb, new_img_xl:new_img_xr] = new_section_image

        return new_image

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

    # try:
    if 1:
        screenCnt = get_document_contour(image)
        # ps_img = four_point_transform(image, screenCnt)

        # bn_img = _convert_in_binary_image(cv2.cvtColor(ps_img, cv2.COLOR_BGR2GRAY), bf=5, bf_v=75, bs=5, c=-5)
        # left, right, top, bottom = _get_stripped_corner_points(
        #     bn_img, max_keep=15, padding=10
        # )
        # ps_img = ps_img[top:bottom, left:right]

        # cv2.imshow("intermediate", ps_img)
        # return ps_img

        return image
    # except Exception as e:
    #     print(e)
    #     return image



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
        # cv2.imshow("output", perspective_img)

        k = cv2.waitKey(0)
        if k == ord("q"):
            break
    
