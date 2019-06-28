import cv2
import numpy as np
#as = alias 
from matplotlib import pyplot as plt

#Range.png : For range comparison 
fname = 'Lenna.png'
Rfname = 'Range.png'
black = np.asarray([0,0,0])
white = np.asarray([255,255,255])

#Default reading method = cv2.IMREAD_COLOR
#Read like [B][G][R]
img = cv2.imread(fname)
Rimg = cv2.imread(Rfname)
width,height = img.shape[:2]
Rwidth,Rheight = Rimg.shape[:2]

#Initialize
Delta = np.zeros_like(img)
Cur = np.zeros_like(img)

#Delta = np.zeros([512,512,3])
#Result = np.zeros([512,512,3])


#Current step's image (Initial = original image)
## Significant 
HighCur = np.array(img)
LowCur = np.array(img)

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
    HSftRight = cv2.warpAffine(HighCur,MR,(height,width))
    HSftLeft = cv2.warpAffine(HighCur,ML,(height,width))
    HSftUp = cv2.warpAffine(HighCur,MU,(height,width))
    HSftDown = cv2.warpAffine(HighCur,MD,(height,width))

    LSftRight = cv2.warpAffine(LowCur,MR,(height,width))
    LSftLeft = cv2.warpAffine(LowCur,ML,(height,width))
    LSftUp = cv2.warpAffine(LowCur,MU,(height,width))
    LSftDown = cv2.warpAffine(LowCur,MD,(height,width))
    
    #Pre-processing (Stride)
    for x in range(0,512):
        HSftRight[x][0] = HSftRight[x][1]
        HSftLeft[x][512-1] = HSftLeft[x][511-1]
        HSftUp[0][x] = HSftUp[1][x]
        HSftDown[512-1][x] = HSftDown[511-1][x]

        LSftRight[x][0] = LSftRight[x][1]
        LSftLeft[x][512-1] = LSftLeft[x][511-1]
        LSftUp[0][x] = LSftUp[1][x]
        LSftDown[512-1][x] = LSftDown[511-1][x]

    #HIgh
    HDelta = HSftRight + HSftLeft + HSftUp + HSftDown -(4*HighCur)
    HighResult = HighCur + (1/200000)*HDelta
    HighCur = HighResult

    #Low
    LDelta = LSftRight + LSftLeft + LSftUp + LSftDown -(4*LowCur)
    LowResult = LowCur + (1/16)*LDelta
    LowCur = LowResult

    ### Smoothing - Heat Equation ###
    for b in range(0,Rheight):
        for a in range(0,Rwidth):
            if(np.array_equal(Rimg[a][b], black)) :
                Cur[a][b]=HighResult[a][b]
            else :
                Cur[a][b]=LowResult[a][b]

if(i==MaxIter-1):
    #print(Cur)
    cv2.imshow('Original',img)
    cv2.imshow('Result',Cur)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite('Result.png',Cur)


