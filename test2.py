from matplotlib import pyplot
import pylab
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import random
import mpld3

fig = pylab.figure()
ax = Axes3D(fig)

v = 80


data = np.genfromtxt('pointcloud1.fuse')
x = data[:,0]
y = data[:,1]
z = data[:,2]
color = data[:,3]

scatter = ax.scatter(x[0::v] ,y[0::v] ,z[0::v], c=color[0::v], cmap='plasma')
pyplot.show()
