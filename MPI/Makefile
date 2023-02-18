EXES = mpitest popp71 popp71scatter popp76 popp71scatterv

all: $(EXES)

% : %.c
	mpicc $< -o $@

# max seems to be 8 on CS1
runtest : mpitest
	mpirun -n 8 ./mpitest

runpopp71 : popp71
	mpirun -n 2 ./popp71

runpopp71scatter : popp71scatter
	mpirun -n 2 ./popp71scatter

runpopp76 : popp76
	mpirun -n 8 ./popp76

clean :
	rm -f $(EXES)
