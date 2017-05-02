from matplotlib import pyplot
import pylab
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import random
import mpld3

fig = pylab.figure()
ax = Axes3D(fig)

v = 100


data = np.genfromtxt('pointcloud1.fuse')
x = data[:,0]
y = data[:,1]
z = data[:,2]

scatter = ax.scatter(x[0::v] ,y[0::v] ,z[0::v] )
pyplot.show()