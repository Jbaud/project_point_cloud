from matplotlib import pyplot
import pylab
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import random
import mpld3

fig = pylab.figure()
ax = Axes3D(fig)

v = 100
threshold = 80


data = np.genfromtxt('pointcloud1.fuse')
x = data[:,0]
y = data[:,1]
z = data[:,2]
color = data[:,3]





x= x[0::v] 
y =y[0::v]
z= z[0::v]
color = color[0::v]

to_delete = []
for index,elt in enumerate(color):
    if color[index] < threshold:
        to_delete.append(index)

new_color = np.delete(color, to_delete)
new_x = np.delete(x, to_delete)
new_y = np.delete(y, to_delete)
new_z = np.delete(z, to_delete)

average_x =  reduce(lambda a, b: a + b, new_x) / len(new_x)
average_y =  reduce(lambda a, b: a + b, new_y) / len(new_y)
average_z =  reduce(lambda a, b: a + b, new_z) / len(new_z)

print average_x
print average_y

scatter = ax.scatter(new_x ,new_y ,new_z, c=new_color, cmap='plasma')
pyplot.show()
