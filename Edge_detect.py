import numpy as np
from Function import *

# ------------------- Edge Detection --------------       
# k=1 sẽ là Prewitt 
# k=2*sqrt(2) là Frei_Chen
# k=2 là Sobel
def Edge_Detection(matrix, k):
    wx=1/(2+k)*np.array([[1,0,-1],[k,0,-k],[1,0,-1]])
    wy=1/(2+k)*np.array([[-1,-k,-1],[0,0,0],[1,k,1]])
    WIDTH=len(matrix[0])
    HEIGHT=len(matrix)
    imgx=np.zeros((HEIGHT, WIDTH))
    imgy=np.zeros((HEIGHT, WIDTH))
    # Đạo hàm theo x và y
    for i in range(1,HEIGHT-1):
        for j in range(1,WIDTH-1):
            f=Matrix_split3x3(matrix,i,j)
            imgx[i][j]=OP_Convolution(f,wx)
            imgy[i][j]=OP_Convolution(f,wy)
    img=np.sqrt(imgx*imgx+imgy*imgy)
    return img

# Edge_Detection theo toán tử Laplace 
def MyLaplacian(matrix):
    laplace=[[1,1,1],[1,-8,1],[1,1,1]]
    WIDTH=len(matrix[0])
    HEIGHT=len(matrix)
    img=np.zeros((HEIGHT, WIDTH))
    # Tích chập laplace
    for i in range(1,HEIGHT-1):
        for j in range(1,WIDTH-1):
            f=Matrix_split3x3(matrix,i,j)
            img[i][j]=OP_Convolution(f,laplace)
    return img