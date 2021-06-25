from functools import wraps
from timeit import default_timer as time
import cv2
from haar_detection import detect_car_haar
import os
import numpy as np
# TODO tweak imports


def measure_time(f):
    """
    decorating function to measure time of function f execution
    :param f: function whose time is measured
    :return: the results of function f and total time of its execution
    """

    # noinspection PyShadowingNames
    @wraps(f)
    def wrapper(*args, **kwargs):
        start = time()
        result = f(*args, **kwargs)
        print(result)
        end = time()
        duration = end - start
        # print('Elapsed time: {} seconds'.format(duration) + ' for function ' + f.__name__)
        if f.__name__ not in results_dict.keys():
            results_dict[f.__name__] = [duration]
        else:
            results_dict[f.__name__].append(duration)
        return result, duration

    return wrapper


def average_results(results_dictionary):
    with open(os.path.join("..", "avg_results"), "a") as f:
        for key in results_dictionary:
            f.write(key + " {}\n".format(np.average(results_dict[key])))
    f.close()


results_dict = {}  # dictionary where keywords are function name, values are times of execution
detect_car_haar = measure_time(detect_car_haar)  # add decorator to imported function

cascade_src = os.path.join("..", "libs", "cars.xml")  # path to trained haar cascade classifier
img_path = os.path.join("..", "data", "VAHA2-1")  # path to processed directory
images = os.listdir(img_path)  # list of images in processed directory

car_cascade = cv2.CascadeClassifier(cascade_src)  # load haar cascade classifier

function_names = ["detect_car_haar"]  # add here your function name you want to evaluate and compare
for image_name in images:
    img = cv2.imread(os.path.join(img_path, image_name))
    # ADD HERE YOUR FUNCTION TO GET A TIME
    for function_name in function_names:
        processed_img = eval("detect_car_haar(img, car_cascade)")
        if function_name == "detect_car_haar":  # haar cascade is needed to be loaded and passed as param (i am lazy)
            processed_img = eval(function_name + "(img, car_cascade)")
        else:
            processed_img = eval(function_name + "(img)")
        cv2.imshow(function_name + " result", processed_img[0])  # np array is returned by function as tuple
        if cv2.waitKey(33) == 27:  # 27 = ESC to quit script
            average_results(results_dict)
            exit()

