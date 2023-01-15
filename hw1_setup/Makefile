CPPFLAGS = -std=c++11 -Wall -Werror -pedantic -ggdb -pthread

hw1_setup : hw1_setup.cpp
	g++ $(CPPFLAGS) $< -o $@

hw1_setup_seq : hw1_setup_seq.cpp
	g++ $(CPPFLAGS) $< -o $@

example : example.cpp ThreadGroup.h
	g++ $(CPPFLAGS) $< -o $@

all : example hw1_setup hw1_setup_seq hw1
	@echo "Made it all!"
