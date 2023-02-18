CPPFLAGS = -std=c++11 -Wall -Werror -pedantic -ggdb -pthread
JAVAFLAGS = -J-Xmx48m
PROGRAMS = general_schwartz

all : $(PROGRAMS)

% : %.cpp GeneralScanSchwartz.h
	g++ $(CPPFLAGS) $< -o $@

clean :
	rm -f $(PROGRAMS)
