import numpy as np
import cv2
DIMDOWN = False
def normalize (x):
    x = np.array(x)
    #return np.exp(x)/sum(np.exp(x))
    sum_x = sum(x)
    if(sum_x==0):
        sum_x = 0.001
    if(DIMDOWN):
        return x/sum_x/40
    else:
        return x/sum_x
weights = [
        np.random.normal(size=(2,3)),
        #np.random.normal(size=(3,2)),
    ]
SIZE = 64
out = np.zeros((SIZE,SIZE,3))
for i in range(SIZE):
    for j in range(SIZE):
        inputx = [i/64,j/64]
        print(inputx)
        out[i][j] = normalize(np.matmul(inputx,weights[0]))
while(True):    
    #out = normalize(out)*2550000
    print(out)
    interpolations=[cv2.INTER_LINEAR,cv2.INTER_AREA,cv2.INTER_NEAREST,cv2.INTER_LANCZOS4]
    #for i in range(SIZE):
    #    for j in range(SIZE):
    #        if(i==j or i-j==1):
    #                out[i][j] = (normalize(np.random.normal(size=(3)))*150)*(1-(i/2-j/2))
    out = cv2.resize(out,(1024,1024),interpolation=interpolations[1])
    cv2.imshow('test',out)
    k = cv2.waitKey(0)
    if(k==ord('q')):
        cv2.destroyAllwindows()
        break
    elif(k==ord("t")):
        with open("weird.txt","w") as file:
            file.write(str(weights[0]))
    elif(k==ord("d")):
        DIMDOWN = True
        out = np.zeros((SIZE,SIZE,3))        
        for i in range(SIZE):
            for j in range(SIZE):
                inputx = [i,j]
                x = normalize(np.matmul(inputx,weights[0]))
                out[i][j] = x
        out = cv2.resize(out,(1024,1024),interpolation=interpolations[1])
        cv2.imshow('test',out)
    elif(k==ord("m")):
        DIMDOWN = False
        out = np.zeros((SIZE,SIZE,3))        
        for i in range(SIZE):
            for j in range(SIZE):
                inputx = [i,j]
                x = normalize(np.matmul(inputx,weights[0]))
                out[i][j] = x
        out = cv2.resize(out,(1024,1024),interpolation=interpolations[1])
        cv2.imshow('test',out)
    elif(k==ord('n')):
        print("hi")
        weights = [
            np.random.normal(size=(2,3)),
            ]
        out = np.zeros((SIZE,SIZE,3))        
        for i in range(SIZE):
            for j in range(SIZE):
                inputx = [i,j]
                x = normalize(np.matmul(inputx,weights[0]))
                out[i][j] = x