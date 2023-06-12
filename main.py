from cv2 import CV_64F, Laplacian
import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
from Color_transformation import *
from Edge_detect import *
from Function import *
from Image_transformation import *
from Smoothing import *
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Image-Processing")
    parser.add_argument('-input', type=str,
                        default='./Input/Lenna.jpg', help='Path of input', )
    parser.add_argument('-output', type=str,
                        default="./Output/", help='Path of output')

    argv = parser.parse_args()

    img_gray = cv2.imread(argv.input, cv2.IMREAD_GRAYSCALE)
    output = argv.output
    ################
    # cv2.IMREAD_COLOR sẽ được giải mã theo ảnh trắng đen (mức xám).
    
    print(img_gray.shape)
    plt.imshow(img_gray, cmap='gray')
    plt.savefig(output+"LENA.png")
    # --------------- Phép biến đổi màu -------------------
    # Biến đổi tuyến tính
    brightness = LinearColor(img_gray, 1, 50)
    plt.imshow(brightness, cmap='gray')
    plt.savefig(output+"MYBRIGHTNESS.png")
    bright_img = cv2.convertScaleAbs(img_gray, alpha=1, beta=50)
    plt.imshow(bright_img, cmap='gray')
    plt.savefig(output+"BRIGHTNESS.png")

    contrast = LinearColor(img_gray, 3/2, 0)
    plt.imshow(contrast, cmap='gray')
    plt.savefig(output+"MYCONTRAST.png")
    contrast_img = cv2.convertScaleAbs(img_gray, alpha=3/2, beta=0)
    plt.imshow(contrast_img, cmap='gray')
    plt.savefig(output+"CONTRAST.png")

    mix = LinearColor(img_gray, 3/2, 50)
    plt.imshow(mix, cmap='gray')
    plt.savefig(output+"MYBRIGHTNESS + MYCONTRAST.png")
    mix_img = cv2.convertScaleAbs(img_gray, alpha=3/2, beta=50)
    plt.imshow(mix_img, cmap='gray')
    plt.savefig(output+"BRIGHTNESS + CONTRAST.png")

    logarith = LogarithmColor(img_gray, 2)
    plt.imshow(logarith, cmap='gray')
    plt.savefig(output+"MYLOGARITH.png")

    exponential = ExponentialColor(img_gray)
    plt.imshow(exponential, cmap='gray')
    plt.savefig(output+"MYEXPONENTIAL.png")


    histogramE = HistogramEqualization(img_gray)
    plt.imshow(histogramE, cmap='gray')
    plt.savefig(output+"MYHISTOGRAM-EQUALIZATION.png")


    plt.subplot(6, 2, 1), plt.imshow(img_gray, cmap='gray')
    plt.title('Original'), plt.xticks([]), plt.yticks([])
    plt.subplot(6, 2, 3), plt.imshow(mix, cmap='gray')
    plt.title('Tự viết'), plt.xticks([]), plt.yticks([])
    plt.subplot(6, 2, 4), plt.imshow(mix_img, cmap='gray')
    plt.title('Hàm hỗ trợ'), plt.xticks([]), plt.yticks([])
    plt.subplot(6, 2, 5), plt.imshow(contrast, cmap='gray')
    plt.subplot(6, 2, 6), plt.imshow(contrast_img, cmap='gray')
    plt.subplot(6, 2, 7), plt.imshow(mix, cmap='gray')
    plt.subplot(6, 2, 8), plt.imshow(mix_img, cmap='gray')
    plt.subplot(6, 2, 9), plt.imshow(logarith, cmap='gray')
    plt.subplot(6, 2, 10), plt.imshow(exponential, cmap='gray')
    plt.subplot(6, 2, 11), plt.imshow(histogramE, cmap='gray')
    plt.title('HistogramEqualization'), plt.xticks([]), plt.yticks([])

    plt.show()

    # --------------- Phép biến đổi hình học -----------
    # a là tập 3 điểm ở ảnh gốc
    # b là tập 3 điểm ở ánh đích
    a = np.float32([[50, 50], [200, 50], [50, 200]])
    b = np.float32([[10, 100], [200, 50], [100, 250]])
    affine = Affine(img_gray, a, b)
    plt.imshow(affine, cmap='gray')
    plt.savefig(output+"MYAFFINE.png")

    plt.subplot(1, 3, 1), plt.imshow(img_gray, cmap='gray')
    plt.title('Original'), plt.xticks([]), plt.yticks([])
    plt.subplot(1, 3, 2), plt.imshow(affine, cmap='gray')
    plt.title('Tự viết'), plt.xticks([]), plt.yticks([])
    M = cv2.getAffineTransform(a, b)
    affine_img = cv2.warpAffine(img_gray, M, img_gray.shape)
    plt.subplot(1, 3, 3), plt.imshow(affine_img, cmap='gray')
    plt.title('Hàm hỗ trợ'), plt.xticks([]), plt.yticks([])
    plt.show()

    # ----------Smoothing Image -----------
    # Toán tử trung bình
    blur = cv2.blur(img_gray, (3, 3))
    plt.imshow(blur, cmap='gray')
    plt.savefig(output+"AVERAGING.png")

    # Toán tử trung bình tự viết
    averaging = Smoothing_Image(img_gray, 1)
    plt.imshow(averaging, cmap='gray')
    plt.savefig(output+"MYAVERAGING.png")

    # Toán tử Gaussian
    blur1 = cv2.GaussianBlur(img_gray, (3, 3), 0)
    plt.imshow(blur, cmap='gray')
    plt.savefig(output+"GAUSSIAN.png")

    # Toán tử Gaussian tự viết
    gaussian = Smoothing_Image(img_gray, 2, 2)
    plt.imshow(gaussian, cmap='gray')
    plt.savefig(output+"MYGAUSSIAN.png")

    # Toán tử Median
    blur2 = cv2.medianBlur(img_gray, 3)
    plt.imshow(gaussian, cmap='gray')
    plt.savefig(output+"MEDIAN.png")

    # Toán tử Median tự viết
    median = Median(img_gray)
    plt.imshow(gaussian, cmap='gray')
    plt.savefig(output+"MYMEDIAN.png")


    plt.subplot(4, 2, 1), plt.imshow(img_gray, cmap='gray')
    plt.title('Original'), plt.xticks([]), plt.yticks([])
    plt.subplot(4, 2, 3), plt.imshow(blur, cmap='gray')
    plt.title('Hàm hỗ trợ'), plt.xticks([]), plt.yticks([])
    plt.subplot(4, 2, 4), plt.imshow(averaging, cmap='gray')
    plt.title('Tự viết'), plt.xticks([]), plt.yticks([])
    plt.subplot(4, 2, 5), plt.imshow(blur1, cmap='gray')
    plt.subplot(4, 2, 6), plt.imshow(gaussian, cmap='gray')
    plt.subplot(4, 2, 7), plt.imshow(blur2, cmap='gray')
    plt.subplot(4, 2, 8), plt.imshow(median, cmap='gray')
    plt.show()


    # ------Edge detection -----------
    # Sobel
    ddepth = cv2.CV_64F  # 64-bit float output
    dx = 1  # đạo hàm theo x
    dy = 1  # đạo hàm theo y
    sobelx = cv2.Sobel(img_gray, ddepth, dx, 0)
    sobely = cv2.Sobel(img_gray, ddepth, 0, dy)
    sobel = np.sqrt(sobelx*sobelx+sobely*sobely)
    plt.imshow(sobel, cmap='gray')
    plt.savefig(output+"SOBEL.png")

    # Sobel tự viết
    mysobel = Edge_Detection(img_gray, 2)
    plt.imshow(mysobel, cmap='gray')
    plt.savefig(output+"MYSOBEL.png")

    # Prewitt tự viết
    myprewitt = Edge_Detection(img_gray, 1)
    plt.imshow(myprewitt, cmap='gray')
    plt.savefig(output+"MYPREWITT.png")

    # Prei-Chen tự viết
    myFrei_Chen = Edge_Detection(img_gray, 2*math.sqrt(2))
    plt.imshow(myFrei_Chen, cmap='gray')
    plt.savefig(output+"MYFREI_CHEN.png")

    # Toán tử Laplacian
    laplacian = cv2.Laplacian(img_gray, cv2.CV_64F)
    plt.imshow(laplacian, cmap='gray')
    plt.savefig(output+"LAPACIAN.png")

    # Toán tử Laplacian tự viết
    mylaplacian = MyLaplacian(img_gray)
    plt.imshow(mylaplacian, cmap='gray')
    plt.savefig(output+"MYLAPACIAN.png")

    plt.subplot(5, 2, 1), plt.imshow(img_gray, cmap='gray')
    plt.title('Original'), plt.xticks([]), plt.yticks([])
    plt.subplot(5, 2, 3), plt.imshow(laplacian, cmap='gray')
    plt.title('Hàm hỗ trợ'), plt.xticks([]), plt.yticks([])
    plt.subplot(5, 2, 4), plt.imshow(mylaplacian, cmap='gray')
    plt.title('Tự viết'), plt.xticks([]), plt.yticks([])
    plt.subplot(5, 2, 5), plt.imshow(sobel, cmap='gray')
    plt.subplot(5, 2, 6), plt.imshow(mysobel, cmap='gray')
    plt.subplot(5, 2, 7), plt.imshow(myprewitt, cmap='gray')
    plt.subplot(5, 2, 9), plt.imshow(myFrei_Chen, cmap='gray')
    plt.show()

    # ----------Smoothing Image -----------
    # Toán tử trung bình
    blur = cv2.blur(img_gray, (3, 3))
    plt.imshow(blur, cmap='gray')
    plt.savefig(output+"AVERAGING.png")

    # Toán tử trung bình tự viết
    averaging = Smoothing_Image(img_gray, 1)
    plt.imshow(averaging, cmap='gray')
    plt.savefig(output+"MYAVERAGING.png")

    # Toán tử Gaussian
    blur1 = cv2.GaussianBlur(img_gray, (3, 3), 0)
    plt.imshow(blur, cmap='gray')
    plt.savefig(output+"GAUSSIAN.png")

    # Toán tử Gaussian tự viết
    gaussian = Smoothing_Image(img_gray, 2, 2)
    plt.imshow(gaussian, cmap='gray')
    plt.savefig(output+"MYGAUSSIAN.png")

    # Toán tử Median
    blur2 = cv2.medianBlur(img_gray, 3)
    plt.imshow(gaussian, cmap='gray')
    plt.savefig(output+"MEDIAN.png")

    # Toán tử Median tự viết
    median = Median(img_gray)
    plt.imshow(gaussian, cmap='gray')
    plt.savefig(output+"MYMEDIAN.png")


    plt.subplot(5, 2, 1), plt.imshow(img_gray, cmap='gray')
    plt.title('Original'), plt.xticks([]), plt.yticks([])
    plt.subplot(5, 2, 3), plt.imshow(blur, cmap='gray')
    plt.title('Hàm hỗ trợ'), plt.xticks([]), plt.yticks([])
    plt.subplot(5, 2, 4), plt.imshow(averaging, cmap='gray')
    plt.title('Tự viết'), plt.xticks([]), plt.yticks([])
    plt.subplot(5, 2, 5), plt.imshow(blur, cmap='gray')
    plt.subplot(5, 2, 6), plt.imshow(mysobel, cmap='gray')
    plt.subplot(5, 2, 7), plt.imshow(Edge_Detection(img_gray, 1), cmap='gray')
    plt.subplot(5, 2, 9), plt.imshow(
        Edge_Detection(img_gray, math.sqrt(2)), cmap='gray')
    plt.show()


    ##
