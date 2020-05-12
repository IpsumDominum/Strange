import numpy as np
import cv2
def normalize (x):
    x = np.array(x)
    sum_x = sum(x)
    if(sum_x==0):
        sum_x = 0.001
    return x/sum_x/10
weights = [
        np.random.normal(size=(4,3)),
        np.zeros((3,2))
        #np.random.normal(size=(3,2)),
    ]

SIZE = 100
prev_out = np.zeros((SIZE,SIZE,3))
prev_out_factor = 1
input_factor = 1
distance_factor = 1
saved_idx = 0
error = np.zeros((SIZE,SIZE,2))
result = cv2.imread("../download.jpeg")
cv2.imshow("mona",result)
OMODE = True
PMODE = True
amount = 0.1
while(True):
    out = np.zeros((SIZE,SIZE,3))
    for i in range(SIZE):
        for j in range(SIZE):
            recur = normalize(np.matmul(prev_out[i][j],weights[1]))            
            inputx = [i/SIZE,j/SIZE,recur[0],recur[1]]
            out[i][j] = normalize(np.matmul(inputx,weights[0]))            
            prev_out[i][j] = out[i][j]
    out = cv2.resize(out,(512,512))
    cv2.imshow('test',out)
    k = cv2.waitKey(1)
    if(k==ord('q')):
        cv2.destroyAllwindows()
        break
    elif(k==ord("t")):
        with open("weird.txt","w") as file:
            file.write(str(weights[0]))
    elif(k==ord('p')):
        if(PMODE):
            PMODE = False
            amount = -0.1
        else:
            PMODE = True
            amount = 0.1
    elif(k==ord('o')):
        if(OMODE):
            OMODE = False
        else:
            OMODE = True
            
    elif(k==ord('1')):
        if(OMODE==True):
            weights[0][0][0] += amount
        else:
            weights[0][2][0] += amount
    elif(k==ord('2')):
        if(OMODE==True):
            weights[0][0][1] += amount
        else:
            weights[0][2][1] += amount

    elif(k==ord('3')):
        if(OMODE==True):
            weights[0][0][2] += amount
        else:
            weights[0][2][2] += amount
    elif(k==ord('4')):
        if(OMODE==True):
            weights[0][1][0] += amount
        else:
            weights[0][3][0] += amount
    elif(k==ord('5')):
        if(OMODE==True):
            weights[0][1][1] += amount
        else:
            weights[0][3][1] += amount
            print(weights[0][3][1])
    elif(k==ord('6')):
        if(OMODE==True):
            weights[0][1][2] += amount
        else:
            weights[0][3][2] += amount
    elif(k==ord('a')):    
        weights[1][0][0] +=0.1
        cv2.imshow("hi",cv2.resize(weights[1],(512,512)))                
    elif(k==ord('s')):    
        weights[1][0][1] +=0.1
        cv2.imshow("hi",cv2.resize(weights[1],(512,512)))                
    elif(k==ord('d')):    
        weights[1][1][0] +=0.1
        cv2.imshow("hi",cv2.resize(weights[1],(512,512)))                
    elif(k==ord('f')):    
        weights[1][1][1] +=0.1
        cv2.imshow("hi",cv2.resize(weights[1],(512,512)))                
    elif(k==ord('g')):    
        weights[1][2][0] +=0.1
        cv2.imshow("hi",cv2.resize(weights[1],(512,512)))                
    elif(k==ord('h')):    
        weights[1][2][1]+=0.1
        cv2.imshow("hi",cv2.resize(weights[1],(512,512)))                
    elif(k==ord('z')):    
        weights[1][0][0] -=0.1
        cv2.imshow("hi",cv2.resize(weights[1],(512,512)))                
    elif(k==ord('x')):    
        weights[1][0][1] -=0.1
        cv2.imshow("hi",cv2.resize(weights[1],(512,512)))                
    elif(k==ord('c')):    
        weights[1][1][0]-=0.1
        cv2.imshow("hi",cv2.resize(weights[1],(512,512)))                
    elif(k==ord('v')):    
        weights[1][1][1] -=0.1
        cv2.imshow("hi",cv2.resize(weights[1],(512,512)))                
    elif(k==ord('b')):    
        weights[1][2][0] -=0.1
        cv2.imshow("hi",cv2.resize(weights[1],(512,512)))                
    elif(k==ord('n')):    
        weights[1][2][1] -=0.1
        cv2.imshow("hi",cv2.resize(weights[1],(512,512)))                
        