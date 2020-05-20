import numpy as np
import cv2

def normalize(x):
    x = np.array(x)
    sum_x = sum(x)
    if(sum_x>0.001):
        return x/sum_x
    else:
        return x

weights = [
    np.random.normal(size=(4,3)),
    np.random.normal(size=(3,2))
]

out = np.zeros((64,64,3))
prev_out = np.zeros((64,64,3))
import time
while(True):
    for i in range(64):
        for j in range(64):
            recur = normalize(np.matmul(prev_out[i][j],weights[1]))
            inputx = [i,j,recur[0],recur[1]]
            out[i][j] = normalize(np.matmul(inputx,weights[0]))
            prev_out[i][j] = out[i][j]
        #out_show = cv2.resize(out,(512,512),interpolation=cv2.INTER_AREA)
        #cv2.imshow("out",out_show)
        #k = cv2.waitKey(1)

    cv2.imshow("out",out_show)

    k = cv2.waitKey(1)
    if(k==ord('n')):
        weights = [
        np.random.normal(size=(4,3)),
        np.random.normal(size=(3,2))
        ]        
        print(weights)
        np.random.seed(int(time.time()))
    elif(k==ord('q')):
        cv2.destroyAllWindows()
        break
