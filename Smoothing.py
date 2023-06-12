import numpy as np
from Function import *
import math

# ------------------------- Làm trơn ảnh -------------------------
# type=1 là sử dụng toán tử trung bình
# type=2 là sử dụng toán tử Gaussian
def Smoothing_Image(matrix, type=2, o=2):
    h=[]
    # Tính hàm h() để tích chập với ma trận của từng điểm ảnh
    if type==1:
        h=1/9*np.array([[1,1,1],[1,1,1],[1,1,1]])
    elif type==2:
        for i in range(-1,2):
            a=[]
            for j in range(-1,2):
                x=(1/(math.sqrt(2*math.pi)*o))*math.exp(-(i*i+j*j)/(2*o*o))
                a.append(x)
            h.append(a)
    h=np.array(h)
    WIDTH=len(matrix[0])
    HEIGHT=len(matrix)
    img=np.zeros((HEIGHT, WIDTH))
    # Tích chập f(i,j) và h
    for i in range(1,HEIGHT-1):
        for j in range(1,WIDTH-1):
            f=Matrix_split3x3(matrix,i,j)
            img[i][j]=OP_Convolution(f,h)
    return img
 
# Làm trơn ảnh theo toán tử Median(Trung vị)
def Median(matrix):
    WIDTH=len(matrix[0])
    HEIGHT=len(matrix)
    img=np.zeros((HEIGHT, WIDTH))
    for i in range(1,HEIGHT-1):
        for j in range(1,WIDTH-1):
            img[i][j]=OP_Median(matrix,i,j)
    return img
            