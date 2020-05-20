import numpy as np
import cv2
def normalize (x):
    x = np.array(x)
    return x/sum(x)
"""
weights = [
        np.random.normal(size=(4,3)),
        np.random.normal(size=(3,2)),
    ]
"""
weights = [
    np.random.rand(6,3),
    np.random.rand(3,2)
]
prev_out = np.zeros((64,64,2))
prev_out_factor = 1
input_factor = 1
distance_factor = 1
error = [0.2,0.2,0.2]
saved_idx = 0
while(True):
    out = np.zeros((64,64,3))
    for i in range(64):
        for j in range(64):
            prev_out[i][j] *= prev_out_factor
            d1 = np.power(32-i,2)
            d2 = np.power(32-j,2)
            inputx = [i*input_factor,j*input_factor,d1*distance_factor,d2*distance_factor,prev_out[i][j][0],prev_out[i][j][1]]
            out[i][j] = normalize(np.matmul(inputx,weights[0]))            
            prev_out[i][j] = normalize(np.matmul(out[i][j],weights[1]))
    out = cv2.resize(out,(512,512))            
    cv2.putText(out,"distance: "+str(distance_factor),(40,40) ,cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0))
    cv2.putText(out,"input: "+str(input_factor),(40,80) ,cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0))
    cv2.putText(out,"recur: "+str(prev_out_factor),(40,120) ,cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0))
    cv2.imshow('test',out)
    k = cv2.waitKey(1)
    if(k==ord('q')):
        cv2.destroyAllwindows()
        break
    elif(k==ord('u')):   
        prev_out_factor += 1
    elif(k==ord('d')):   
        prev_out_factor -= 1
    elif(k==ord('l')):   
        input_factor += 1
    elif(k==ord('r')):     
        input_factor -= 1
    elif(k==ord('a')):
        distance_factor +=1
    elif(k==ord('z')):
        distance_factor -=0.1
    elif(k==ord('s')):
        np.save("saved"+str(saved_idx)+".npy",np.array(weights))
    elif(k==ord('v')):
        weights = np.load("saved0.npy")
    elif(k==ord('n')):
        """
        weights = [
                np.random.normal(size=(4,3)),
                np.random.normal(size=(3,2)),
            ]        
        """
        weights = [
         -1 +np.random.rand(6,3)*3,
         -1 +np.random.rand(3,2)*3
        ]   

