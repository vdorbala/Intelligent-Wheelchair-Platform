from __future__ import division
from math import sqrt
import sys
import numpy as np

def cuberoot(x):
    if 0<=x: return x**(1./3.)
    return -(-x)**(1./3.)

def main():
    # img = cv2.imread("/home/vdorbala/ICRA/Captured_Images/Captured22/10.png")
    # w,img1 = classical(img)
    # print ("w original is {}".format(w))
    lambda_1 = 100
    w = sys.argv[1:]
    w = float(w[0])
    print ("Angular Velocity is {}".format(w))

    C = np.divide((-1*w), 2*lambda_1)
    print(C)

    D = sqrt((np.square(w)/(4*(np.square(lambda_1)))) + (1/27))
    print (D)

    A = cuberoot(C + D)
    B = cuberoot(C - D)

    solution = A + B
    print(solution)


if __name__ == '__main__':
	main()