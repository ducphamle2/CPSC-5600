CPPFLAGS =

cuda-test: cuda-test.cu
	nvcc $(CPPFLAGS) $< -o $@

hello-world: hello-world.cu
	nvcc $(CPPFLAGS) $< -o $@

all : cuda-test hello-world
	@echo "Made it all!"
