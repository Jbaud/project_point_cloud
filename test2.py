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

#fig = pylab.figure()
#ax = Axes3D(fig)


def is_outlier(points, thresh=3.5):
    """
    Returns a boolean array with True if points are outliers and False 
    otherwise.

    Parameters:
    -----------
        points : An numobservations by numdimensions array of observations
        thresh : The modified z-score to use as a threshold. Observations with
            a modified z-score (based on the median absolute deviation) greater
            than this value will be classified as outliers.

    Returns:
    --------
        mask : A numobservations-length boolean array.

    References:
    ----------
        Boris Iglewicz and David Hoaglin (1993), "Volume 16: How to Detect and
        Handle Outliers", The ASQC Basic References in Quality Control:
        Statistical Techniques, Edward F. Mykytka, Ph.D., Editor. 
    """
    if len(points.shape) == 1:
        points = points[:,None]
    median = np.median(points, axis=0)
    diff = np.sum((points - median)**2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation

    return modified_z_score > thresh


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

print z

to_delete = []
for index,elt in enumerate(color):
	if color[index] < threshold:
	   	to_delete.append(index)

print len(to_delete)
new_color = np.delete(color, to_delete)
new_x = np.delete(x, to_delete)
new_y = np.delete(y, to_delete)
new_z = np.delete(z, to_delete)

#average_x =  reduce(lambda a, b: a + b, new_x) / len(new_x)
#average_y =  reduce(lambda a, b: a + b, new_y) / len(new_y)
#average_z =  reduce(lambda a, b: a + b, new_z) / len(new_z)

#print average_x
#print average_y

# save to disk as csv 
c = np.column_stack((new_x,new_y,new_z,new_color))
np.savetxt('/Users/Jules/Downloads/final_project_data/test.txt', c, delimiter=' ', fmt='%s')

#sort by first column
d  = c[c[:,0].argsort()]

# looking for a sharp diff in alt -> means we changed side 
for index,row in enumerate(d):
   if (d[index][3] - d[index-1][3] >70 ) and index > 0:
	part12 = d[0:index].copy()
	part1 = d[index:-1].copy()

part1 = np.delete(part1, [3], axis=1)
part2 = np.delete(part1, [3], axis=1)		

np.savetxt('/Users/Jules/Downloads/final_project_data/part1.txt', part1, delimiter=' ,', fmt='%s')
np.savetxt('/Users/Jules/Downloads/final_project_data/part2.txt', part2, delimiter=' ,', fmt='%s')

#def reject_outliers(data, m):
#    return data[abs(data - np.mean(data)) < m * np.std(data)]

#part1 = reject_outliers(part1,6)
model_robust, inliers = ransac(part1, LineModelND, min_samples=2,
                               residual_threshold=0.00001, max_trials=1000)
outliers = inliers == False



fig = pyplot.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(part1[inliers][:, 0], part1[inliers][:, 1], part1[inliers][:, 2], c='b',
           marker='o', label='Inlier data')
ax.scatter(part1[outliers][:, 0], part1[outliers][:, 1], part1[outliers][:, 2], c='r',
           marker='o', label='Outlier data')
average_x =  reduce(lambda a, b: a + b, part1[inliers][:, 0]) / len(part1[inliers][:, 0])
average_y =  reduce(lambda a, b: a + b, part1[inliers][:, 1]) / len(part1[inliers][:, 1])
average_z =  np.min(new_z)

ax.scatter(average_x, average_y, average_z, c='g',
           marker='o', label='mean data')

ax.legend(loc='lower left')
pyplot.show()



'''
data = np.c_[part1[inliers][:, 0],part1[inliers][:, 1],part1[inliers][:, 2]]


# regular grid covering the domain of the data
mn = np.min(data, axis=0)
mx = np.max(data, axis=0)
X,Y = np.meshgrid(np.linspace(mn[0], mx[0], 20), np.linspace(mn[1], mx[1], 20))

order = 1    # 1: linear, 2: quadratic
if order == 1:
    # best-fit linear plane
    A = np.c_[data[:,0], data[:,1], np.ones(data.shape[0])]
    C,_,_,_ = scipy.linalg.lstsq(A, data[:,2])    # coefficients
    
    # evaluate it on grid
    Z = C[0]*X + C[1]*Y + C[2]
    
    # or expressed using matrix/vector product
    #Z = np.dot(np.c_[XX, YY, np.ones(XX.shape)], C).reshape(X.shape)

elif order == 2:
    # best-fit quadratic curve
    A = np.c_[np.ones(data.shape[0]), data[:,:2], np.prod(data[:,:2], axis=1), data[:,:2]**2]
    C,_,_,_ = scipy.linalg.lstsq(A, data[:,2])
    
    # evaluate it on a grid
    Z = np.dot(np.c_[np.ones(XX.shape), XX, YY, XX*YY, XX**2, YY**2], C).reshape(X.shape)

'''
# plot points and fitted surface

"""
fig = pyplot.figure()
#ax = fig.gca(111,projection='3d')
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, alpha=0.2)
#ax.scatter(new_x, new_y, new_z, c='b', s=50)
ax.scatter(part1[inliers][:, 0], part1[inliers][:, 1], part1[inliers][:, 2], c='r',
           marker='o', label='Inlier data')
pyplot.xlabel('X')
pyplot.ylabel('Y')
ax.set_zlabel('Z')
ax.axis('equal')
ax.axis('tight')
pyplot.show()
"""


#scatter = ax.scatter(new_x ,new_y ,new_z, c=new_color, cmap='plasma')

#only shows one wall 


#scatter = ax.scatter(part2[:,0],part2[:,1],part2[:,2], c=part2[:,3], cmap='plasma')
#pyplot.show()
