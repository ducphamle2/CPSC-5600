CPPFLAGS = -std=c++11 -Wall -Werror -pedantic -ggdb -pthread
JAVAFLAGS = -J-Xmx48m
PROGRAMS = bitonic bitonic_loops bitonic_rec
CLASSES = HW1.class Codec.class

java_example : $(CLASSES)
	@echo "To run type: java HW1"

all : $(PROGRAMS) java_example

% : %.cpp
	g++ $(CPPFLAGS) $< -o $@

%.class : %.java
	javac $(JAVAFLAGS) $<

clean :
	rm -f $(PROGRAMS) $(CLASSES)

# JAVAFLAGS - 48mb size machine - change memory size. Needed in CS1
# How to run java file: java <name>.class or java <name>
# thread.join() - wait until thread is finished
# Synchronous queue - queue de luu tru data cho cac threads su dung. Khi a thread inserts data vo queue thi se bi block, baoh co thread khac su dung queue thi se dequeue ra luon va dua cho thread do su dung. take() la de dequeue va su dung data. put() - 

# Ladner-Fisher prefix sum la oblivious boi vi no k quan tam array data la gi, no luon cong phan tu 0 va 1, 1 va 2, 10 va 11 voi nhau, chi can dua cho array cung size thi value la gi cung deu lam nhu vay het => oblivious

# Quicksort: dua vao data de chon pivot => se sort cac phan tu khac nhau neu nhu pivot khac nhau. Pivot thi dua vao element value

# Bitonic merge: 

# Bitonic sort complexity: log2(n) levels * O(n /2 * log2(n)) where n / 2 in O(n/2 * log2(n)) is number of maximum comparators in a level, and log2(n) is num of sub columns in a level