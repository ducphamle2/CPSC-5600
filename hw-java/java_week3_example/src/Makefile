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
