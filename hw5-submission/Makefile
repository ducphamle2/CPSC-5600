CPPFLAGS = -std=c++20 -Wall -Werror -pedantic -ggdb -pthread
PROGRAMS = extra-credit

# This Makefile includes commands to build & run the extra credit files. It uses and additional -pthread for an attempt to calculate euclid distance in parallel.
# type: 'make all' to build all
# type: 'make run-extra-credit' to run the extra credit task
# The submission also includes two MNIST test datasets: 't10k-images-idx3-ubyte' & 't10k-labels-idx1-ubyte'. Without them, there will be errors when running the extra credit program.

all : $(PROGRAMS)

MnistKmeans.o : Mnist.cpp Mnist.h
	mpic++ $(CPPFLAGS) $< -c -o $@

extra-credit.o: extra-credit.cpp Mnist.h MnistKmeans.h KMeansMPI.h
	mpic++ $(CPPFLAGS) $< -c -o $@

extra-credit: extra-credit.o MnistKmeans.o
	mpic++ $(CPPFLAGS) extra-credit.o MnistKmeans.o -o $@

run-extra-credit : extra-credit
	mpirun -n 32 ./extra-credit


# MnistKmeans.o : Mnist.cpp Mnist.h
# 	mpic++ $(CPPFLAGS) $< -c -o $@ -I /usr/local/include/opencv4 -lopencv_core -lopencv_videoio -lopencv_highgui -lopencv_imgproc

# extra-credit.o: extra-credit.cpp Mnist.h MnistKmeans.h KMeansMPI.h
# 	mpic++ $(CPPFLAGS) $< -c -o $@ -I /usr/local/include/opencv4 -lopencv_core -lopencv_videoio -lopencv_highgui -lopencv_imgproc

# extra-credit: extra-credit.o MnistKmeans.o
# 	mpic++ $(CPPFLAGS) extra-credit.o MnistKmeans.o -o $@ -I /usr/local/include/opencv4 -lopencv_core -lopencv_videoio -lopencv_highgui -lopencv_imgproc

clean :
	rm -f $(PROGRAMS)
