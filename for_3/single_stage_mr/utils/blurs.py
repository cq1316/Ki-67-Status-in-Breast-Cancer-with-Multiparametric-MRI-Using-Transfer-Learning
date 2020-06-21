import cv2


def CLAHE_blur(src_img, size):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(size, size))
    dst = clahe.apply(src_img)
    return dst


def Sobel_blur(src_img):
    x = cv2.Sobel(src_img, cv2.CV_16S, 1, 0)
    y = cv2.Sobel(src_img, cv2.CV_16S, 0, 1)
    absX = cv2.convertScaleAbs(x)  # 转回uint8
    absY = cv2.convertScaleAbs(y)
    dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
    return dst
