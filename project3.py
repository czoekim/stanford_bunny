# Project 3
# Carmen Cheng (kcheng) | Zoe Kim (ckim)
# On my honor, I have not given, nor received, nor witnessed any unauthorized assistance on this work.
# We worked on this assignment and referred to
# https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_geometric_transformations/py_geometric_transformations.html
# https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html
# https://docs.opencv.org/2.4/modules/imgproc/doc/miscellaneous_transformations.html
# http://www.bogotobogo.com/python/OpenCV_Python/python_opencv3_Image_Global_Thresholding_Adaptive_Thresholding_Otsus_Binarization_Segmentations.php

# Images:
# https://pixabay.com/en/seamless-pattern-background-tile-1904165/
# https://pixabay.com/en/sheet-rip-shiny-metal-embossed-861640/
# http://res.freestockphotos.biz/pictures/9/9343-a-cute-orange-kitten-isolated-on-a-white-background-pv.jpg
# https://c1.staticflickr.com/3/2433/3776502226_cfbec6a216_b.jpg
# https://cdn.pixabay.com/photo/2016/08/09/18/01/pineapple-1581281_960_720.jpg
import cv2
import numpy as np
import scipy as sp

img_gray = cv2.imread('cat.jpg', 0)
metal = cv2.imread("metal2.jpg", 0)
metal = cv2.resize(metal, (img_gray.shape[1], img_gray.shape[0]))

def removeNoise(image):
    """ This function applies Gaussian Blur to the image to remove significant
    noise in the image that may skew the outline/edge detection.

    Args:
        image (numpy.ndarray): A grayscale image represented in a numpy array.

    Returns:
        blur_img (numpy.ndarray): Blurred image.
    """
    blur_img = cv2.GaussianBlur(image, (5,5),0)
    return blur_img

def apply_binary(image):
    """ This function calls the Adaptive Threshold function to isolate the image's outline
    and transform the outline into a binary image. As one of its parameter the adaptiveThreshold
    function uses Adaptive Gaussian thresholding, which uses a weighted sum as a threshold
    for each pixel.

    Args:
        image (numpy.ndarray): A grayscale image represented in a numpy array.

    Returns:
        img_out (numpy.ndarray): A binary outline of the subject

    """
    img_out = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                                                        cv2.THRESH_BINARY, 5, 2)
    return img_out

def make_mask(img, threshold):
    """ This function creates a binary mask of a subject on a white background.

    Args:
        img (numpy.ndarray): A grayscale image represented in a numpy array.
        threshold (int): The threshold value.

    Returns:
        mask (numpy.ndarray): A binary mask of the subject.
    """
    mask = img.copy()
    rows, cols = img.shape
    for r in range(rows):
        for c in range(cols):
            if img[r][c] < threshold:
                mask[r][c] = 1
            else:
                mask[r][c] = 0

    kernel = np.ones((5,5),np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask

def alpha_blend(image1, image2, alpha):
    """ This function applies alpha blending to two image.

    Args:
        image1 (numpy.ndarray): A colored image represented in a numpy array.
        image2 (numpy.ndarray): A colored image represented in a numpy array.
        alpha (double): Value of alpha.

    Returns:
        img (numpy.ndarray): The blended image.
    """
    img = np.zeros(image1.shape, dtype=np.int32)
    if len(image1.shape) > 2:
        rows, cols, c = image1.shape
        for r in range(rows):
            for c in range(cols):
                img[r][c] = alpha * image1[r][c][:3] + (1-alpha) * image2[r][c][:3]
    else:
        rows, cols = image1.shape
        for r in range(rows):
            for c in range(cols):
                img[r][c] = alpha * image1[r][c] + (1-alpha) * image2[r][c]
    return img.astype(np.uint8)

def gamma_correction(img, correction):
    """ This function adjusts the contrast of a grayscale input image.

    Args:
        img (numpy.ndarray): A grayscale image represented in a numpy array.
        correction (float): Threshold level.

    Returns:
        numpy.ndarray: The adjusted image.
    """

    img = img / 255
    img = cv2.pow(img, correction)
    return np.uint8(img * 255)

def make_metallic(metal, outline, mask, alpha=.9, gamma=1.5):
    """ This function uses alpha blending to overlay metallic textures over the
    masked image, then increases its contrast with gamma correction.

    Args:
        metal (numpy.ndarray): A grayscale image of the overlay represented in a numpy array.
        outline (numpy.ndarray): The outline of the subject created with binary filter.
        mask (numpy.ndarray): The mask of the subject.
        alpha (float): The value of alpha.
        gamma (float): The value of gamma.

    Returns:
        res (numpy.ndarray): The resulting image.
    """
    res = metal * mask
    outline = outline * mask
    res = alpha_blend(res, outline, alpha)
    res = gamma_correction(res, gamma)
    return res

def generate_shadow(img, mask, x=0, y=5):
    """ This function uses Gaussian Blur to generate a shadow of the masked image,
    translates it downwards, then combines the two images with the shadow on the
    bottom "layer".

    Args:
        img (numpy.ndarray): A grayscale image represented in a numpy array.
        mask (numpy.ndarray): A binary mask of the subject.
        x (int): Units of translation in the x-axis.
        y (int): Units of translation in the y-axis.

    Returns:
        final (numpy.ndarray): The combined image of the subject and its shadow.
    """
    shadow = np.zeros(mask.shape, mask.dtype)
    rows, cols = mask.shape
    for r in range(rows):
        for c in range(cols):
            if mask[r][c] == 0:
                shadow[r][c] = 255
            else:
                shadow[r][c] = 30
    shadow = cv2.GaussianBlur(shadow, (25, 25), 0)
    matrix = np.float32([[1,0,x],[0,1,y]])
    shadow = cv2.warpAffine(shadow, matrix, (shadow.shape[1], shadow.shape[0]))

    final = np.zeros(mask.shape, mask.dtype) + 255
    final += shadow
    final[final > 251] = 255
    for r in range(rows):
        for c in range(cols):
            if mask[r][c] == 1:
                final[r][c] = img[r][c]

    return final

def create_canvas(img):
    """ This function creates a canvas of default size 600 x 800. The subject is
    scaled to a width of 300 and placed in the center of the canvas. It is then
    converted to RGB.

    Args:
        img (numpy.ndarray): A grayscale image represented in a numpy array.

    Returns:
        canvas (numpy.ndarray): The 600 x 800 canvas with the subject centered.
    """
    canvas = np.zeros((600, 800), dtype=np.uint8) + 255
    height = int(img.shape[0] / (img.shape[1] / 300))
    obj = cv2.resize(img, (300, height))
    start_row = (600//2) - (height//2)
    start_col = (800//2) - (300//2)
    canvas[start_row:start_row+height, start_col:start_col+300] += obj
    canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
    return canvas

def create_color_overlay(b, g, r):
    """ This function generates a custom color overlay.

    Args:
        b (int): The blue value.
        g (int): The green value.
        r (int): The red value.

    Returns:
        overlay (numpy.ndarray): A color overlay.
    """
    overlay = np.zeros((600, 800, 3), dtype=np.uint8)
    overlay[:,:] = (b, g, r)
    return overlay

blurred = removeNoise(img_gray)
outline = apply_binary(blurred)

mask = make_mask(blurred, 232)
subj = make_metallic(metal, outline, mask, alpha=.8, gamma=2.0)
with_shadow = generate_shadow(subj, mask)


final = create_canvas(with_shadow)
overlay = create_color_overlay(0, 180, 232)

gamber = alpha_blend(final, overlay, .7)

cv2.imshow("Gambered", gamber)
cv2.imwrite("kitty.jpg", gamber)
cv2.waitKey(0)
cv2.destroyAllWindows()
