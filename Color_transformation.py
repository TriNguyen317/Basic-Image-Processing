import numpy as np
import math
# --------------- Phép biến đổi màu -------------------
# 1. Biến đổi tuyến tính 
# img[i][j]= a*f[i][j] +b
def LinearColor(matrix, a=0, b=0):
    WIDTH=len(matrix[0])
    HEIGHT=len(matrix)
    img=np.zeros((HEIGHT, WIDTH))
    for i in range(HEIGHT):
        for j in range(WIDTH):
            img[i][j]= 255 if (a* matrix[i][j] +b)>=256 else (a* matrix[i][j] +b)
    return img
# 2. Phép biến đổi màu phi tuyến
# Hàm logarithm: img[i][j]=c*log(f[i][j])
def LogarithmColor(matrix, c):
    WIDTH=len(matrix[0])
    HEIGHT=len(matrix)
    img=np.zeros((HEIGHT, WIDTH))
    for i in range(HEIGHT):
        for j in range(WIDTH):
            img[i][j]= 255 if (c* math.log(matrix[i][j]))>=256 else c* math.log(matrix[i][j])
    return img
# Hàm mũ cơ số e: img[i][j] = e^f[i][j]
def ExponentialColor(matrix):
    WIDTH=len(matrix[0])
    HEIGHT=len(matrix)
    img=np.zeros((HEIGHT, WIDTH))
    for i in range(HEIGHT):
        for j in range(WIDTH):
            img[i][j]= 255 if (math.exp(matrix[i][j]/45.53))>=256 else math.exp(matrix[i][j]/45.53)
    return img
# 3. Phép biến đổi màu bằng Cân bằng lược đồ xám
def HistogramEqualization(matrix):
    WIDTH=len(matrix[0])
    HEIGHT=len(matrix)
    size=WIDTH*HEIGHT
    img=np.zeros((HEIGHT, WIDTH))
    h=np.zeros(256)
    for i in range(HEIGHT):
        for j in range(WIDTH):
            h[matrix[i][j]]+=1
    t=np.zeros(256)
    for i in range(1,256):
        t[i]=t[i-1]+h[i]
    for i in range(1,256):
        t[i]=round((255/size)*t[i])
    for i in range(HEIGHT):
        for j in range(WIDTH):
            img[i][j]=t[matrix[i][j]]
    return img

# 4. Phép biến đổi màu bằng Đặc tả lược đồ xám 
def HistogramSpecification(matrix, g):
    WIDTH=len(matrix[0])
    HEIGHT=len(matrix)
    size=WIDTH*HEIGHT
    h=np.zeros(256);
    img=np.zeros((HEIGHT, WIDTH))
    for i in range(HEIGHT):
        for j in range(WIDTH):
            h[matrix[i][j]]+=1
    t=np.zeros(256)
    for i in range(1,256):
        t[i]=t[i-1]+h[i]
        t[i]=round((255/size)*t[i])
        
    G=np.zeros(256)
    for i in range(1,256):
        G[i]=G[i-1]+g[i]
        G[i]=round((255/size)*G[i])
        
    for i in range(HEIGHT):
        for j in range(WIDTH):
            img[i][j]=G[t[matrix[i][j]]]
    return img