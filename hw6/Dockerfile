FROM nvidia/cuda:11.0.3-devel-ubuntu20.04

RUN apt -y update && apt -y upgrade && apt install -y nano

# install c++
RUN apt install -y build-essential gcc-9 g++-9 cpp-9 make

# configure C++
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 100 --slave /usr/bin/g++ g++ /usr/bin/g++-9 --slave /usr/bin/gcov gcov /usr/bin/gcov-9

# steps: need to match cuda docker version with local cuda version. type: nvidia-smi to check cuda version locally, then choose nvidia cuda image that matches. Otherwise there will be an error: Warning CUDA device error.
# then need to follow: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker
# Cuda extension on VSCode: https://developer.nvidia.com/nsight-visual-studio-code-edition
# to run a helloworld: https://cuda-tutorial.readthedocs.io/en/latest/tutorials/tutorial01/