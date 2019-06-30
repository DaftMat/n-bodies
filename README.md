# n-bodies
Enjoy this lil naive n-bodies !

# Dependencies
You'll need python3 with mpi4py, numpy, scipy and matplotlib in order to run this code :
```
$ sudo apt-get install python3.6
$ sudo apt-get install python3-pip
$ sudo pip3 install mpi4py
$ sudo pip3 install numpy
$ sudo pip3 install scipy
$ sudo pip3 install matplotlib
```

# Run program
Here's the command :
```
mpirun -np <nb_threads> python3 n-bodies.py <nb_bodies> <nb_steps>
```
With `<nb_step>` the number of images that must be calculated before leaving the program
**Warning** : it will have bad behavior if `<nb_bodies>` can't be devided by `<nb_threads>`

A good example of running this program :
```
mpirun -np 4 python3 n-bodies.py 12 1000
```
