CPPFLAGS = -std=c++20 -Wall -Werror -pedantic -ggdb -pthread
PROGRAMS = kmean_color_test hw5 extra-credit

all : $(PROGRAMS)

Color.o : Color.cpp Color.h
	mpic++ $(CPPFLAGS) $< -c -o $@

MnistKmeans.o : Mnist.cpp Mnist.h
	mpic++ $(CPPFLAGS) $< -c -o $@ -I /usr/local/include/opencv4 -lopencv_core -lopencv_videoio -lopencv_highgui -lopencv_imgproc

kmean_color_test.o : kmean_color_test.cpp Color.h ColorKMeans.h KMeans.h
	mpic++ $(CPPFLAGS) $< -c -o $@

kmean_color_test : kmean_color_test.o Color.o
	mpic++ $(CPPFLAGS) kmean_color_test.o Color.o -o $@

run_sequential : kmean_color_test
	./kmean_color_test

hw5.o : hw5.cpp Color.h ColorKMeansMPI.h KMeansMPI.h
	mpic++ $(CPPFLAGS) $< -c -o $@

hw5 : hw5.o Color.o
	mpic++ $(CPPFLAGS) hw5.o Color.o -o $@

test : test.o
	mpic++ $(CPPFLAGS) test.o -o $@

read-mnist: read-mnist.cpp
	g++ $(CPPFLAGS) $< -o $@ -I /usr/local/include/opencv4 -lopencv_core -lopencv_videoio -lopencv_highgui -lopencv_imgproc

extra-credit.o: extra-credit.cpp Mnist.h MnistKmeans.h KMeansMPI.h KMeans.h
	mpic++ $(CPPFLAGS) $< -c -o $@ -I /usr/local/include/opencv4 -lopencv_core -lopencv_videoio -lopencv_highgui -lopencv_imgproc

extra-credit: extra-credit.o MnistKmeans.o
	mpic++ $(CPPFLAGS) extra-credit.o MnistKmeans.o -o $@ -I /usr/local/include/opencv4 -lopencv_core -lopencv_videoio -lopencv_highgui -lopencv_imgproc

run_hw5 : hw5
	mpirun -n 2 ./hw5

valgrind : hw5
	mpirun -n 2 valgrind -s --leak-check=full --show-leak-kinds=all --suppressions=valgrind.supp --gen-suppressions=all ./hw5

bigger_test : hw5
	mpirun -n 10 ./hw5

clean :
	rm -f $(PROGRAMS) Color.o kmean_color_test.o hw5.o 
