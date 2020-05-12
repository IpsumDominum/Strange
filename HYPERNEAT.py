import cv2
import numpy as np
DARKNESS = 100
VIEWMODE = False
DARKNESSMULT = 0.001
def normalize(x):     
    #return x/sum(x)
    return x,x/sum(x)
    #return x/sum(x)/10
def sigmoid(x):
    return 1/1+np.exp(-x)
def neuronet(x,weights,prev):
    d1 = np.power(32-x[1],2)
    d2 = np.power(32-x[0],2)
    x = x*np.matmul(prev,weights[0])
    inputs = [x[0],x[1],1,np.sqrt(d1+d2),d1,d2]
    x = np.matmul(inputs,weights[2]) #+ weights[2][0] #+ np.matmul(prev,weights[1])
    for layer in range(3,len(weights)):
        x = np.matmul(x,weights[layer])
    x,x2 = normalize(x)    
    return x,x2
def get_population(size):
    population = []
    for i in range(size[0]):
        population.append([])
        for j in range(size[1]):
            population[i].append([
                    np.random.normal(size=(3,2)),#Recursive connection1                    
                    np.random.normal(size=(1,32)),#Bias
                    np.random.normal(size=(6,32)),#FirstLayer
                    np.random.normal(size=(32,3)),
                    ])
    return population

def evolve_next(chosen_ones):
    new_pop = []
    for n in range(3):
        new_pop.append([])
        for k in range(3):            
            if(np.random.rand()<0.8):
                if(len(chosen_ones)>1):
                    parents = np.random.choice(len(chosen_ones),size=2,replace=False)
                else:
                    parents = (0,0)
                baby = crossover(chosen_ones[parents[0]],chosen_ones[parents[1]])
            else:
                parents = get_population((1,2))
                baby = crossover(parents[0][0],parents[0][1])
            if(np.random.rand()<0.8):
                baby = mutate(baby)
            new_pop[n].append(baby)
    return new_pop
            
def crossover(weights1,weights2):
    baby = weights1.copy()
    for layer in range(len(weights1)):
        for i in range(len(weights1[layer])):
            baby[layer][i][:len(weights1[layer][i])//2] = weights2[layer][i][:len(weights1[layer][i])//2]                
    return baby
def mutate(weights1):
    baby = weights1.copy()
    for layer in range(len(weights1)):
        for i in range(len(weights1[layer])):
            for j in range(len(weights1[layer][i])):
                if(np.random.rand()<0.4):
                    baby[layer][i][j] += np.random.normal()
    return baby

#---HYPERPARAMS---
POPSIZE = (3,3)
SIZE = 50
display_multiplier = 4
generation_num = 0
population = get_population(POPSIZE)
prev_output = np.zeros((POPSIZE[0]*POPSIZE[1],SIZE,SIZE,3))
while(True):
    chosen_ones = {k:None for k in range(0,9)}
    all_chosen = []
    #-------Propagate-----
    out = np.zeros((SIZE*POPSIZE[0],SIZE*POPSIZE[1],3))
    tempout = np.zeros((SIZE*POPSIZE[0],SIZE*POPSIZE[1],3))
    for n in range(POPSIZE[0]):
        for k in range(POPSIZE[1]):
            res = np.zeros((SIZE,SIZE,3))
            res2 = np.zeros((SIZE,SIZE,3))
            weights = population[n][k]
            prev = prev_output[n*POPSIZE[1]+k]
            for i in range(SIZE):
                for j in range(SIZE):
                    res[i][j],res2[i][j] = neuronet([i,j],weights,prev[i][j])
            all_chosen.append(weights)
            prev_output[n*POPSIZE[1]+k] = res2
            out[n*SIZE:(n+1)*SIZE,k*SIZE:(k+1)*SIZE,:] = res
            tempout[n*SIZE:(n+1)*SIZE,k*SIZE:(k+1)*SIZE,:]= res2
    generation_num +=1
    WRITE = True
   
    if(WRITE):
        out = cv2.resize(out,(out.shape[0]*display_multiplier,out.shape[1]*display_multiplier))
        tempout = cv2.resize(tempout,(tempout.shape[0]*display_multiplier,tempout.shape[1]*display_multiplier))
        for n in range(POPSIZE[0]):
            for k in range(POPSIZE[1]):
                cv2.putText(out,str(n*POPSIZE[1]+k+1), (k*SIZE*display_multiplier+display_multiplier*10,n*SIZE*display_multiplier+display_multiplier*10), cv2.FONT_HERSHEY_SIMPLEX, display_multiplier//4, (255,255,255))
    #out = np.reshape(out,(SIZE*3,SIZE*3,3))
    instructions = np.zeros((512,512,3))
    cv2.putText(instructions,"INSTRUCTIONS",(0,30) ,cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,255))
    cv2.putText(instructions,"USE 1-9 to select individuals,",(0,60) ,cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255))
    cv2.putText(instructions,"press n to evolve",(0,120) ,cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255))
    cv2.putText(instructions,"press q to quit ",(0,150) ,cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255))    
    cv2.putText(instructions,"GENERATION:{}".format(generation_num),(0,200) ,cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255))
    cv2.putText(instructions,"DARKNESS : {}".format(DARKNESS),(0,230) ,cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255))
    cv2.putText(instructions,"DARKNESS : {}".format(DARKNESS),(0,230) ,cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255))
    cv2.putText(instructions,"k(1),d(100) to inc darkness",(0,260) ,cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255))
    cv2.putText(instructions,"j(1),u(100) to dec darkness",(0,290) ,cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255))
    cv2.putText(instructions,"DARKNESSMULT : {}".format(DARKNESSMULT),(0,320) ,cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255))
    cv2.putText(instructions,"m(down),l(up) to scale dmult",(0,350) ,cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255))
    cv2.putText(instructions,"VIEWMODE : {}".format(VIEWMODE),(0,380) ,cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255))
    if(VIEWMODE==True):
        cv2.imshow('hi',tempout/DARKNESS*DARKNESSMULT)    
    else:
        cv2.imshow('hi',out/DARKNESS*DARKNESSMULT)
    cv2.imshow('instructions',instructions)
    rect_map = {}
    for i in range(POPSIZE[0]):
        for j in range(POPSIZE[1]):
            rect_map[i*POPSIZE[1]+j] = (j,i)
    chosen = {k:False for k in range(0,9)}


    #---Non evolving mode---
    #k = cv2.waitKey(1)
    #continue
    while(True):
        k = cv2.waitKey(0)        
        if(k in [ord(str(num))for num in range(1,10)]):
            #K is number
            if(chosen[k-48-1]):
                chosen_ones[k-48-1] = None
                chosen[k-48-1] = False
                block_coord = rect_map[k-48-1]
                x1 = block_coord[0]*SIZE
                y1 = block_coord[1]*SIZE
                x2 = x1 + SIZE
                y2 = y1 +SIZE
                x1 *= display_multiplier
                y1 *= display_multiplier
                x2 *= display_multiplier
                y2 *= display_multiplier
                cv2.rectangle(out, (x1,y1),(x2,y2), (0,0,0), 2)
            else:
                chosen[k-48-1] = True
                chosen_ones[k-48-1] = all_chosen[k-48-1]
                block_coord = rect_map[k-48-1]
                x1 = block_coord[0]*SIZE
                y1 = block_coord[1]*SIZE
                x2 = x1 + SIZE
                y2 = y1 +SIZE
                x1 *= display_multiplier
                y1 *= display_multiplier
                x2 *= display_multiplier
                y2 *= display_multiplier
                cv2.rectangle(out, (x1,y1),(x2,y2), (0,255,0), 2)
            cv2.imshow('hi',out)
        elif(k ==ord('q')):
            cv2.destroyAllWindows()
            exit()
        elif(k == ord('n')):
            checked_chosen_ones = [chosen_ones[key] for key in chosen_ones if chosen_ones[key]!=None]
            if(len(checked_chosen_ones)<=0):
                checked_chosen_ones = [chose[0] for chose in population]
                population = evolve_next(checked_chosen_ones)
                prev_output = np.zeros((POPSIZE[0]*POPSIZE[1],SIZE,SIZE,3))
                break
                #cv2.putText(instructions,"Choose one first".format(generation_num),(0,230) ,cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0))
                #cv2.imshow('instructions',instructions)
            else:
                break
        elif(k==ord('j')):
            #Tune up brightness
            DARKNESS -=1            
        elif(k==ord('k')):
            #Tune down
            DARKNESS +=1            
        elif(k==ord('u')):
            DARKNESS -=100            
        elif(k==ord('d')):
            DARKNESS +=100
            instructions = np.zeros((512,512,3))            
        elif(k==ord('r')):
            DARKNESS = 100            
        elif(k==ord('l')):
            DARKNESSMULT *= 10
        elif(k==ord('m')):     
            DARKNESSMULT *= 0.1
        elif(k==ord('v')):
            if(VIEWMODE==True):
                VIEWMODE = False
            else:
                VIEWMODE = True
        else:
            break
        instructions = np.zeros((512,512,3))
        cv2.putText(instructions,"INSTRUCTIONS",(0,30) ,cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,255))
        cv2.putText(instructions,"USE 1-9 to select individuals,",(0,60) ,cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255))
        cv2.putText(instructions,"press n to evolve",(0,120) ,cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255))
        cv2.putText(instructions,"press q to quit ",(0,150) ,cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255))    
        cv2.putText(instructions,"GENERATION:{}".format(generation_num),(0,200) ,cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255))
        cv2.putText(instructions,"DARKNESS : {}".format(DARKNESS),(0,230) ,cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255))
        cv2.putText(instructions,"k(1),d(100) to inc darkness",(0,260) ,cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255))
        cv2.putText(instructions,"j(1),u(100) to dec darkness",(0,290) ,cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255))
        cv2.putText(instructions,"DARKNESSMULT : {}".format(DARKNESSMULT),(0,320) ,cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255))
        cv2.putText(instructions,"m(down),l(up) to scale dmult",(0,350) ,cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255))
        cv2.putText(instructions,"VIEWMODE : {}".format(VIEWMODE),(0,380) ,cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255))
        if(VIEWMODE==True):            
            cv2.imshow('hi',tempout/DARKNESS*DARKNESSMULT)    
        else:
            cv2.imshow('hi',out/DARKNESS*DARKNESSMULT)
        cv2.imshow('instructions',instructions)

        





