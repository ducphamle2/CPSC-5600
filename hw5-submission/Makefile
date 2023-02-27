CPPFLAGS = -std=c++20 -Wall -Werror -pedantic -ggdb -pthread
PROGRAMS = hw5 extra-credit

# This Makefile includes commands to build & run the extra credit files. It uses and additional -pthread for an attempt to calculate euclid distance in parallel.
# type: 'make all' to build all
# type: 'make run_extra_credit' to run the extra credit task

all : $(PROGRAMS)

Color.o : Color.cpp Color.h
	mpic++ $(CPPFLAGS) $< -c -o $@

MnistKmeans.o : Mnist.cpp Mnist.h
	mpic++ $(CPPFLAGS) $< -c -o $@

extra-credit.o: extra-credit.cpp Mnist.h MnistKmeans.h KMeansMPI.h
	mpic++ $(CPPFLAGS) $< -c -o $@

extra-credit: extra-credit.o MnistKmeans.o
	mpic++ $(CPPFLAGS) extra-credit.o MnistKmeans.o -o $@

run_extra_credit : extra-credit
	mpirun -n 10 ./extra-credit


# MnistKmeans.o : Mnist.cpp Mnist.h
# 	mpic++ $(CPPFLAGS) $< -c -o $@ -I /usr/local/include/opencv4 -lopencv_core -lopencv_videoio -lopencv_highgui -lopencv_imgproc

# extra-credit.o: extra-credit.cpp Mnist.h MnistKmeans.h KMeansMPI.h
# 	mpic++ $(CPPFLAGS) $< -c -o $@ -I /usr/local/include/opencv4 -lopencv_core -lopencv_videoio -lopencv_highgui -lopencv_imgproc

# extra-credit: extra-credit.o MnistKmeans.o
# 	mpic++ $(CPPFLAGS) extra-credit.o MnistKmeans.o -o $@ -I /usr/local/include/opencv4 -lopencv_core -lopencv_videoio -lopencv_highgui -lopencv_imgproc

hw5.o : hw5.cpp Color.h ColorKMeansMPI.h KMeansMPI.h
	mpic++ $(CPPFLAGS) $< -c -o $@

hw5 : hw5.o Color.o
	mpic++ $(CPPFLAGS) hw5.o Color.o -o $@

run_hw5 : hw5
	mpirun -n 2 ./hw5

valgrind : hw5
	mpirun -n 2 valgrind -s --leak-check=full --show-leak-kinds=all --suppressions=valgrind.supp --gen-suppressions=all ./hw5

bigger_test : hw5
	mpirun -n 10 ./hw5

clean :
	rm -f $(PROGRAMS) Color.o hw5.o 
