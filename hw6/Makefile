CPPFLAGS =

hello-world: hello-world.cu
	nvcc $(CPPFLAGS) $< -o $@

hw6: hw6.cu
	nvcc $(CPPFLAGS) $< -o $@

run_hw6: hw6
	nvprof ./hw6

all : hello-world hw6
	@echo "Made it all!"
