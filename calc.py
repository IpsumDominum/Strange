import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Button
with open("weird3.txt","r") as file:
    w = np.array(eval(file.read()))
with open("weird.txt","r") as file:
    wr = np.array(eval(file.read()))
def normalize (x):
    x = np.array(x)
    sum_x = sum(x)
    if(sum_x==0):
        sum_x = 0.001
    return x/sum_x/10
prev_val = np.zeros((1,1,3))
val = np.zeros((1,1,3))

idx = 0
val_history1 = []
val_history2 = []
val_history3 = []
class glob:
    x = 1
    y = 1
for i in range(10000):    
    rec = normalize(np.matmul(prev_val[0][0],wr))
    inputx = [glob.x,glob.y,rec[0],rec[1]]
    val[0][0]= normalize(np.matmul(inputx,w))
    prev_val[0][0] = val[0][0]
    val_history1.append(val[0][0][0])
def update_line(num,val_history1,l):
    prev_val = np.zeros((1,1,3))
    val = np.zeros((1,1,3))
    val_history1 = []
    for i in range(10000):
        rec = normalize(np.matmul(prev_val[0][0],wr))        
        inputx = [glob.x,glob.y,rec[0],rec[1]]
        val[0][0]= normalize(np.matmul(inputx,w))
        prev_val[0][0] = val[0][0]        
        val_history1.append(val[0][0][0])
    l.set_data(range(1000),val_history1[num:num+1000])
    return l,

class Index(object):
    def addx(self, event):
        glob.x +=1        
        print(glob.x)
    def addy(self, event):
        glob.x -=1
        print(glob.x)

fig = plt.figure()
l, = plt.plot(range(1000),val_history1[:1000])    
axprev = plt.axes([0.7, 0.05, 0.1, 0.075])
axnext = plt.axes([0.81, 0.05, 0.1, 0.075])
callback = Index()
bnext = Button(axnext, 'addx')
bnext.on_clicked(callback.addx)
bprev = Button(axprev, 'addy')
bprev.on_clicked(callback.addy)
ax1_ani = animation.FuncAnimation(fig, update_line, 1000, fargs=(val_history1,l),
                                   interval=1, blit=True)
plt.show()
