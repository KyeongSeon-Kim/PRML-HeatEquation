import cv2
import numpy as np
#as = alias 
from matplotlib import pyplot as plt

fname = 'Lenna.png'

#Default reading method = cv2.IMREAD_COLOR
#Read like [B][G][R]
img = cv2.imread(fname)
width,height = img.shape[:2]

#Current step's image (Initial = original image)
Cur = img

# M=translation array
# Right/Left/Up/Down shifting
MR = np.float32([[1,0,1],[0,1,0]])
ML = np.float32([[1,0,-1],[0,1,0]])
MU = np.float32([[1,0,0],[0,1,-1]])
MD = np.float32([[1,0,0],[0,1,1]])

MaxIter=128;

#Smoothing Iter
for i in range(0,MaxIter):
    print("Iteration ", i,"st")
    ### Image Shifting ###
    SftRight = cv2.warpAffine(Cur,MR,(height,width))
    SftLeft = cv2.warpAffine(Cur,ML,(height,width))
    SftUp = cv2.warpAffine(Cur,MU,(height,width))
    SftDown = cv2.warpAffine(Cur,MD,(height,width))
    
    #Pre-processing (Stride)
    for x in range(0,512):
        SftRight[x][0] = SftRight[x][1]
        SftLeft[x][512-1] = SftLeft[x][511-1]
        SftUp[0][x] = SftUp[1][x]
        SftDown[512-1][x] = SftDown[511-1][x]

    ### Smoothing - Heat Equation ###
    Delta = SftRight + SftLeft + SftUp + SftDown -(4*Cur)
    Result = Cur + (1/16)*Delta
    #Result = cv2.normalize(Result, None, 255,0, cv2.NORM_MINMAX, cv2.CV_8UC1)
    Cur = Result

    if(i==MaxIter-1):

        print(Result)
        #cv2.imshow('Original',img)
        #cv2.imshow('Result',Result)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        cv2.imwrite('Result.png',Result)

'''
    ### Showing ### 
    if(maxIter==0 or maxIter==1 or maxIter==2 or maxIter==4 or maxIter==8 or maxIter==16 or maxIter==32 or maxIter==64 or maxIter==128):
        cv2.imshow('Original',img)
        cv2.imshow('Result',Cur)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
'''
        

