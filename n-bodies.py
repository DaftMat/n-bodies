#mpirun -n 4 python3 n-bodies.py 12 1000
from mpi4py import MPI
import sys
import math
import random
import numpy as np
import matplotlib.image as mpimg
from scipy import misc
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import time

# split a vector "x" in "size" part, each of them having "n" elements
def split(x, n, size):
	return [x[n*i:n*(i+1)] for i in range(size)]

# unsplit a list x composed of lists
def unsplit(x):
	y = []
	n = len(x)
	for i in range(n):
		for j in range(len(x[i])):
			y.append(x[i][j])
	return y

solarmass=1.98892e30

def circlev(rx, ry):
	r2=math.sqrt(rx*rx+ry*ry)
	numerator=(6.67e-11)*1e6*solarmass
	return math.sqrt(numerator/r2)

# from http://physics.princeton.edu/~fpretori/Nbody/code.htm
class Data_item:
	
	def __init__(self, id, positionx, positiony, speedx, speedy, weight):
		self.id = id
		self.positionx = positionx
		self.positiony = positiony
		self.weight = weight
		
		if positionx==0 and positiony==0:    # the center of the world, very heavy one...
			self.speedx = 0
			self.speedy = 0
		else:
			if speedx==0 and speedy==0:			# initial values
				magv=circlev(positionx, positiony)
				absangle = math.atan(math.fabs(positiony/positionx))
				thetav= math.pi/2-absangle
				phiv = random.uniform(0,1)*math.pi
				self.speedx = -1*math.copysign(1, positiony)*math.cos(thetav)*magv
				self.speedy = math.copysign(1, positionx)*math.sin(thetav)*magv
				#Orient a random 2D circular orbit
				if (random.uniform(0,1) <=.5):
					self.speedx=-self.speedx
					self.speedy=-self.speedy
			else:
				self.speedx = speedx
				self.speedy = speedy
	
	def __str__(self):
		return "ID="+str(self.id)+" POS=("+str(self.positionx)+","+str(self.positiony)+") SPEED=("+str(self.speedx)+","+str(self.speedy)+") WEIGHT="+str(self.weight)

def display(m, l):
	for i in range(len(l)):
		print("PROC"+str(rank)+":"+m+"-"+str(l[i]))

def displayPlot(d):
	plt.gcf().clear()			# to remove to see the traces of the particules...
	plt.axis((-1e17,1e17,-1e17,1e17))
	xx = [ d[i].positionx  for i in range(len(d)) ]
	yy = [ d[i].positiony  for i in range(len(d)) ]
	plt.plot(xx, yy, 'ro')
	plt.draw()
	plt.pause(0.00001)			# in order to see something otherwise too fast...


def interaction(i, j):
	dist = math.sqrt( (j.positionx-i.positionx)*(j.positionx-i.positionx) +  (j.positiony-i.positiony)*(j.positiony-i.positiony) )
	if i==j:
		return (0,0)
	g = 6.673e-11
	factor = g * i.weight * j.weight / (dist*dist+3e4*3e4)
	return factor * (j.positionx-i.positionx) / dist, factor * (j.positiony-i.positiony) / dist

def update(d, f):
	dt = 1e11
	vx = d.speedx + dt * f[0]/d.weight
	vy = d.speedy + dt * f[1]/d.weight
	px = d.positionx + dt * vx
	py = d.positiony + dt * vy
	return Data_item(id=d.id, positionx=px, positiony=py, speedx=vx, speedy=vy, weight=d.weight)


def init_world(n):
	data = [ Data_item(id=i, positionx=1e18*math.exp(-1.8)*(.5-random.uniform(0,1)), positiony=1e18*math.exp(-1.8)*(.5-random.uniform(0,1)), speedx=0, speedy=0, weight=(random.uniform(0,1)*solarmass*10+1e20)) for i in range(n-1)]
	data.append( Data_item(id=nbbodies-1, positionx=0, positiony=0, speedx=0, speedy=0, weight=1e6*solarmass))
	return data

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
num_threads = comm.Get_size()

nbbodies = int(sys.argv[1])
NBSTEPS = int(sys.argv[2])

bodies = []
splitted_bodies = []

if rank == 0:
	plt.draw()
	plt.show(block=False)
	bodies = init_world(nbbodies)
	splitted_bodies = split(bodies, int(nbbodies/num_threads), num_threads)

#Only need to call the init_world on one thread, the bcast to the others.
bodies = comm.bcast(bodies, root=0)
#Modifications will be made on the local_bodies, while they'll be reading on global bodies.
local_bodies = comm.scatter(splitted_bodies, root=0)
nbbodies_local = len(local_bodies)

force = [[0,0] for i in range(nbbodies_local)]

# here to start the code...
for t in range(NBSTEPS):
	for i in range(nbbodies_local):
		force[i] = [0,0]
		for j in range(nbbodies):
			f = interaction(bodies[i+nbbodies_local*rank], bodies[j])
			force[i][0]=force[i][0]+f[0]
			force[i][1]=force[i][1]+f[1]
	
	#bodies update
	comm.barrier()
	for i in range(nbbodies_local):
		local_bodies[i] = update(local_bodies[i], force[i])

	comm.barrier()
	splitted_bodies = comm.allgather(local_bodies)
	comm.barrier()
	bodies = unsplit(splitted_bodies)

	comm.barrier()
	if rank == 0:
		displayPlot(bodies)
