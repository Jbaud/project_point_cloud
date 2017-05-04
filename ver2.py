from matplotlib import pyplot
import pylab
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import random
import mpld3
import scipy.optimize
import functools
from sklearn.preprocessing import normalize
import scipy.linalg
from scipy.ndimage.measurements import label
from skimage.measure import LineModelND, ransac
from haversine import haversine

def normal(point,normal):

    # a plane is a*x+b*y+c*z+d=0
    # [a,b,c] is the normal. Thus, we have to calculate
    # d and we're set
    d = -point.dot(normal)

    # create x,y
    xx, yy = np.meshgrid(range(10), range(10))

    # calculate corresponding z
    z = (-normal[0] * xx - normal[1] * yy - d) * 1. /normal[2]
    return xx,yy,z

v = 100
threshold = 50


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

print len(to_delete)
new_color = np.delete(color, to_delete)
new_x = np.delete(x, to_delete)
new_y = np.delete(y, to_delete)
new_z = np.delete(z, to_delete)

#sort by first column
d  = c[c[:,0].argsort()]

# looking for a sharp diff in alt -> means we changed side
#left and right are the two sides on the road 
for index,row in enumerate(d):
   if (d[index][3] - d[index-1][3] >70 ) and index > 0:
    print "at"
    print index
    left = d[:index]
    right = d[index:]

print d.shape
print left.shape
print right.shape
print " here"
print  str(left[30][0]) + "  " + str(left[30][1])
print  str(right[30][0]) + " " + str(right[30][1])
print haversine( (left[10][0],left[10][1]) , (right[30][0],right[30][1]))*1000

fig = pyplot.figure()
ax = fig.add_subplot(111, projection='3d')

model_robust, inliers = ransac(left, LineModelND, min_samples=2,
                               residual_threshold=0.00001, max_trials=1000)
outliers = inliers == False

average_x_1 =  reduce(lambda a, b: a + b, left[inliers][:, 0]) / len(left[inliers][:, 0])
average_y_1 =  reduce(lambda a, b: a + b, left[inliers][:, 1]) / len(left[inliers][:, 1])
average_z_1 =  np.min(new_z)

model_robust, inliers2 = ransac(right, LineModelND, min_samples=2,
                               residual_threshold=0.00001, max_trials=1000)
outliers = inliers == False

average_x_2 =  reduce(lambda a, b: a + b, right[inliers2][:, 0]) / len(right[inliers2][:, 0])
average_y_2 =  reduce(lambda a, b: a + b, right[inliers2][:, 1]) / len(right[inliers2][:, 1])
average_z_2 =  np.min(new_z)

ax.scatter(average_x_2, average_y_2, average_z_2, c='b',
           marker='o', label='mean data2')

ax.plot(   [average_x_1,average_x_2], [ average_y_1,average_y_2] , [average_z_1,average_z_2])
ax.legend(loc='lower left')

pyplot.show()

a =  (average_x_1,average_y_2)
b =  (average_x_2,average_y_2)
print haversine(a, b) * 1000
print " in meter"
print a 
print b