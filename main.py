import numpy as np
import cv2 as cv
from skimage.filters import threshold_local, threshold_niblack, threshold_sauvola
import sys
from matplotlib import pyplot as plt

sharpen1_kernel = np.array([[1,-2,1],[-2,5,-2],[1,-2,1]])
sharpen2_kernel = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])

def b_niblack(img):
    return img > threshold_niblack(img, window_size=7, k=0.1)

def b_sauvola(img):
    return img > threshold_sauvola(img, window_size=15, k=0.2)

def b_adaptive_mean(img):
    return img > threshold_local(img, 15, 'mean')

def test_threshold(img):
    fig, axarr = plt.subplots(2, 2)
    axarr[0,0].imshow(img)
    axarr[0,0].axis('off')
    axarr[0,0].set_title('Original')
    axarr[0,1].imshow(b_niblack(img))
    axarr[0,1].axis('off')
    axarr[0,1].set_title('Niblack')
    axarr[1,0].imshow(b_sauvola(img))
    axarr[1,0].axis('off')
    axarr[1,0].set_title('Sauvola')
    axarr[1,1].imshow(b_adaptive_mean(img))
    axarr[1,1].axis('off')
    axarr[1,1].set_title('Adaptive')
    fig.suptitle('Адаптивная и локальная пороговая обработка')
    plt.show()

def test_sharpen(img):
    fig, axarr = plt.subplots(3)
    axarr[0].axis('off')
    axarr[0].imshow(img)
    axarr[0].set_title('Original')
    axarr[1].axis('off')
    axarr[1].imshow(cv.filter2D(img, -1, sharpen1_kernel))
    axarr[1].set_title('Kernel 1')
    axarr[2].axis('off')
    axarr[2].imshow(cv.filter2D(img, -1, sharpen2_kernel))
    axarr[2].set_title('Kernel 2')
    fig.suptitle('Увеличение резкости')
    plt.show()

def main():
    arg = sys.argv[1]
    name = sys.argv[2]
    img = cv.imread(name)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    if arg == 'sharpen':
        test_sharpen(img)
    elif arg == 'threshold':
        test_threshold(gray)

if __name__ == "__main__":
    main()
