import cv2
import os
import numpy as np


# set ROI
right_top = (110, 395)
dx = (1000, 350)
scale_position = ((223, 584), (521, 577), (570, 617), (249, 625))



img_path = os.path.join("..", "data", "VAHA1")  # path to processed directory
images = os.listdir(img_path)  # list of images in processed directory

kernel = np.ones((21, 21), np.uint8)

background = cv2.imread(os.path.join("..", "data", "background_test_vaha2.jpg"))
background = cv2.cvtColor(background, cv2.COLOR_RGB2GRAY)


background_subtractor = cv2.createBackgroundSubtractorMOG2()  # check settings
# zkusit entropii
# klasifikovat kontury
# 2550 motorka
# 1750 sum, 4120
wierd_img = 1300
for idx, image in enumerate(images[wierd_img:10000]):
    original_image = cv2.imread(os.path.join(img_path, image))
    original_image = cv2.blur(original_image, (3, 3))  # 3,3
    subtracted_image = background_subtractor.apply(original_image)
    eroded = cv2.erode(subtracted_image, kernel, cv2.BORDER_REFLECT)

    kernel = np.ones((3, 3), np.uint8)
    #eroded = cv2.morphologyEx(eroded, cv2.MORPH_CLOSE, kernel)
    #eroded = cv2.erode(subtracted_image, kernel, cv2.BORDER_REFLECT)
    eroded = cv2.dilate(eroded, kernel, iterations=1)
    # eroded = cv2.GaussianBlur(eroded, (5, 5), 0)

    ret, eroded = cv2.threshold(eroded, 200, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(eroded, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    eroded = cv2.cvtColor(eroded, cv2.COLOR_GRAY2BGR)

    cv2.line(eroded, scale_position[0], scale_position[1], (0, 255, 0), 3)
    cv2.line(eroded, scale_position[1], scale_position[2], (0, 255, 0), 3)
    cv2.line(eroded, scale_position[2], scale_position[3], (0, 255, 0), 3)
    cv2.line(eroded, scale_position[3], scale_position[0], (0, 255, 0), 3)

    for contour in contours:
        if cv2.contourArea(contour) > 800:
            M = cv2.moments(contour)
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.drawMarker(original_image, (cx, cy), (255, 0, 0), markerType=cv2.MARKER_STAR, thickness=5)
            cv2.drawContours(eroded, [contour], 0, (255, 0, 0), 3)
    # cv2.imshow("subtracted - eroded", eroded)
    cv2.putText(eroded, image, (right_top[1] + 350, right_top[0] + 300), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2,
                cv2.LINE_AA)
    cv2.putText(eroded, str(idx),  (right_top[1] + 640, right_top[0] + 300), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (155, 150, 150), 2, cv2.LINE_AA)
    cv2.imshow("Subtracted - eroded", eroded[right_top[1]:right_top[1] + dx[1], right_top[0]:right_top[0] + dx[0]])
    cv2.imshow("Original", original_image[right_top[1]:right_top[1] + dx[1], right_top[0]:right_top[0] + dx[0]])
    pressed_key = cv2.waitKey(5)
    if pressed_key == 27:
        break
