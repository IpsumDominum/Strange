import numpy as np
import cv2

def normalize(x):
    sum_x = sum(x) 
    if(sum_x==0):
        sum_x = 0.001
    return x/sum_x/10

weights = [
    np.random.normal(size=(4,3)),
    np.random.normal(size=(3,2))
    ]
SIZE = 64
out = np.zeros((SIZE,SIZE,3))
prev_out = np.zeros((SIZE,SIZE,3))
while True:    
    for i in range(SIZE):
        for j in range(SIZE):
            rec = normalize(np.matmul(prev_out[i][j],weights[1]))
            inputx = [i,j,rec[0],rec[1]]
            out[i][j] = normalize(np.matmul(inputx,weights[0]))
            prev_out[i][j] = out[i][j]
    #out_reshaped = cv2.resize(out,(512,512),interpolation=cv2.INTER_AREA)
    #cv2.imshow('out',out_reshaped)
    cv2.imshow('out',out)
    k = cv2.waitKey(1)
    if(k==ord('q')):
        cv2.destroyAllWindows()
    elif(k==ord('n')):
        weights = [
        np.random.normal(size=(4,3)),
        np.random.normal(size=(3,2))
        ]        
