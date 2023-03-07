CPPFLAGS =

hello-world: hello-world.cu
	nvcc $(CPPFLAGS) $< -o $@

hw6: hw6.cu
	nvcc $(CPPFLAGS) $< -o $@

hw6_scan: hw6-scan.cu
	nvcc $(CPPFLAGS) $< -o $@

run_hw6: hw6
	nvprof ./hw6

run_hw6_scan: hw6_scan
	nvprof --device-buffer-size 1024 ./hw6_scan

all : hello-world hw6 hw6_scan
	@echo "Made it all!"
