import numpy as np
import cv2
# -------------- Phép biến đổi hình học -------------
# 1. Biến đổi vị trí điểm ảnh
# Dạng song tuyến tính theo pp gán giá tri màu điểm ảnh
# Phép biến đổi affine theo pp gán giá trị màu điểm ảnh
def Affine(matrix,souce,des,type=1):
    WIDTH=len(matrix[0])
    HEIGHT=len(matrix)
    img=np.zeros((HEIGHT, WIDTH))
    # Suy từ điểm ảnh gốc ra điểm ảnh đích
    # g(x,y)=a*f(x,y)+b
    # Nhưng làm vậy sẽ bị những lỗ trống trên ảnh => Suy từ ảnh kết quả ra ảnh gốc rồi thực hiện nội suy gán giá trị màu điểm ảnh
    # Tính ma trận biến đổi suy từ ảnh kết quả về ảnh gốc
    M = cv2.getAffineTransform(des,souce)
    # Lưu vào a, b để thực hiện 
    a=np.array([[M[0][0],M[0][1]],[M[1][0],M[1][1]]])
    b=np.array([M[0][2],M[1][2]])
    if type==1:
        for i in range(HEIGHT):
            for j in range(WIDTH):
                # Giá trị màu được tính dựa vào phép nội suy người láng giềng gần nhất
                n=i-b[0]
                m=j-b[1]
                x= round(a[0][0]*i+a[0][1]*j -b[0])
                y= round(a[1][0]*i +a[1][1]*j - b[1])
                if x>=HEIGHT or x<=-HEIGHT or y>=WIDTH or y<=-WIDTH:
                    continue
                img[i][j]=matrix[x][y]
    #img=cv2.warpAffine(matrix, M, (HEIGHT,WIDTH))
    # else:
    #     for i in range(HEIGHT):
    #         for j in range(WIDTH):
    #             n=i-b[0]
    #             m=j-b[1]
    #             x= a[0][0]*n+a[0][1]*m
    #             y= a[1][0]*n +a[1][1]*m
    #             l=round(x)
    #             k=round(y)
    #             a1=x-l
    #             b1=y-k
    #             if l+1>=HEIGHT or l+1<=-HEIGHT or k+1>=WIDTH or k+1<=-WIDTH:
    #                 continue;
    #             img[i][j]=round((matrix[l+1][k]-matrix[l][k])*a1+(matrix[l][k+1]-matrix[l][k])*b1+(matrix[l+1][k+1]+matrix[l][k]-matrix[l][k+1]-matrix[l+1][k])*a1*b1);
    return img