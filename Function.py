import numpy as np

# Toán tử convolution 
def OP_Convolution(matrix1, matrix2):
    width=len(matrix1[0])
    height=len(matrix1)
    sum=0
    for i in range(height):
        for j in range(width):
            sum+=matrix1[i][j]*matrix2[width-1-i][height-1-j]
    return sum

# Lấy 8 điểm lân cận của điểm ảnh thành ma trận 3x3
def Matrix_split3x3(matrix,x,y):
    mchild=[]
    for i in range(-1,2):
        a=[]
        for j in range(-1,2):
            a.append(matrix[x+i][y+j])
        mchild.append(a)
    return np.array(mchild)

# Toán tử Median (trung vị)
def OP_Median(matrix,x,y):
    mchild=[]
    for i in range(-1,2):
        for j in range(-1,2):
            mchild.append(matrix[x+i][y+j])
    mchild=np.array(mchild)
    mchild=np.sort(mchild)
    return mchild[4]



