import cv2
import os


def detect_car_haar(img, car_cascade):
    """
    function to detect cars via haar cascade clasifier
    (see more at https://github.com/jeremy-shannon/CarND-Vehicle-Detection)
    :param img: image obtained by opencv function imread
    :param car_cascade: loaded cascade classifier
    :return: image with marked cars
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cars = car_cascade.detectMultiScale(gray, 1.1, 1)
    for (x, y, w, h) in cars:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
    return img


# # random stuff
# cascade_src = os.path.join("..", "libs", "cars.xml")
# img_path = os.path.join("..", "data", "VAHA2-1")
# images = os.listdir(img_path)
#
# print(cv2.data.haarcascades)
# car_cascade = cv2.CascadeClassifier(cascade_src)
#
# for image_name in images:
#     img = cv2.imread(os.path.join(img_path, image_name))
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     cars = car_cascade.detectMultiScale(gray, 1.1, 1)
#     for (x, y, w, h) in cars:
#         cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
#     cv2.imshow('videoss', img)
#     if cv2.waitKey(33) == 27:  # 27 = ESC
#         break
#
