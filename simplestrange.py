import numpy as np
import cv2
def normalize (x):
    x = np.array(x)
    #return np.exp(x)/sum(np.exp(x))
    sum_x = sum(x)
    if(sum_x==0):
        sum_x = 0.001
    return x/sum_x/5
weights = [
        np.random.normal(size=(2,3)),
        #np.random.normal(size=(3,2)),
    ]

SIZE = 100
while(True):
    out = np.zeros((SIZE,SIZE,3))
    for i in range(SIZE):
        for j in range(SIZE):
            inputx = [i/SIZE,j/SIZE]
            out[i][j] = normalize(np.matmul(inputx,weights[0]))
    out = cv2.resize(out,(512,512))
    cv2.imshow('test',out)
    k = cv2.waitKey(1)
    if(k==ord('q')):
        cv2.destroyAllwindows()
        break
    elif(k==ord("t")):
        with open("weird.txt","w") as file:
            file.write(str(weights[0]))
    elif(k==ord('n')):
        print("hi")
        weights = [np.random.normal(size=(2,3))]
