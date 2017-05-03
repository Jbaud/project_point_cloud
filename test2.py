from matplotlib import pyplot
import pylab
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import random
import mpld3


def PCA(data, correlation = False, sort = True):
	""" Applies Principal Component Analysis to the data

	Parameters
	----------        
	data: array
		The array containing the data. The array must have NxM dimensions, where each
		of the N rows represents a different individual record and each of the M columns
		represents a different variable recorded for that individual record.
			array([
			[V11, ... , V1m],
			...,
			[Vn1, ... , Vnm]])

	correlation(Optional) : bool
			Set the type of matrix to be computed (see Notes):
				If True compute the correlation matrix.
				If False(Default) compute the covariance matrix. 

	sort(Optional) : bool
			Set the order that the eigenvalues/vectors will have
				If True(Default) they will be sorted (from higher value to less).
				If False they won't.   
	Returns
	-------
	eigenvalues: (1,M) array
		The eigenvalues of the corresponding matrix.

	eigenvector: (M,M) array
		The eigenvectors of the corresponding matrix.

	Notes
	-----
	The correlation matrix is a better choice when there are different magnitudes
	representing the M variables. Use covariance matrix in other cases.
	"""

	mean = np.mean(data, axis=0)

	data_adjust = data - mean

	#: the data is transposed due to np.cov/corrcoef syntax
	if correlation:

		matrix = np.corrcoef(data_adjust.T)

	else:
		matrix = np.cov(data_adjust.T) 

	eigenvalues, eigenvectors = np.linalg.eig(matrix)

	if sort:
		#: sort eigenvalues and eigenvectors
		sort = eigenvalues.argsort()[::-1]
		eigenvalues = eigenvalues[sort]
		eigenvectors = eigenvectors[:,sort]

	return eigenvalues, eigenvectors

def best_fitting_plane(points, equation=False):
	""" Computes the best fitting plane of the given points

	Parameters
	----------        
	points: array
		The x,y,z coordinates corresponding to the points from which we want
		to define the best fitting plane. Expected format:
			array([
			[x1,y1,z1],
			...,
			[xn,yn,zn]])

	equation(Optional) : bool
			Set the oputput plane format:
				If True return the a,b,c,d coefficients of the plane.
				If False(Default) return 1 Point and 1 Normal vector.    
	Returns
	-------
	a, b, c, d : float
		The coefficients solving the plane equation.

	or

	point, normal: array
		The plane defined by 1 Point and 1 Normal vector. With format:
		array([Px,Py,Pz]), array([Nx,Ny,Nz])

	"""

	w, v = PCA(points)

	#: the normal of the plane is the last eigenvector
	normal = v[:,2]

	#: get a point from the plane
	point = np.mean(points, axis=0)


	if equation:
		a, b, c = normal
		d = -(np.dot(normal, point))
		return a, b, c, d

	else:
		return point, normal   

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
	part1 = d[0:index].copy()
	part2 = d[index:-1].copy()

#part1 = np.delete(part1, [3], axis=1)
#point1,normal1 =  best_fitting_plane(part1)


#print point1
#print normal1

#scatter = ax.scatter(new_x ,new_y ,new_z, c=new_color, cmap='plasma')
scatter = ax.scatter(part1[:,0],part1[:,1],part1[:,2], c=part1[:,3], cmap='plasma')
pyplot.show()
