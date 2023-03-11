1. NVIDIA card enabled server: ssh dpham6@csp239.cslab.seattleu.edu. Need VPN if you're not in the Seattleu network
2. Tricks to work with non power of 2 - padding with zeroes for scan, and padding inf for bitonic sort
latancy of this scan version is still logn

do one block at a time, and figure out reduction of the first block leaf
The next block leaf with each element added with the reduction

for example, the first block has reduction of 1024, then the next block, each index + 1024

2nd tier combines all the leaf blocks below, each index is the reduction of a leaf below => check image in iphone

bitonic loop: the smallest i loop is going to be inside of the kernel, each index is a thread

k & j loop are handled by the host when threads get bigger than 1024

Blocks are threads in the previous way of learning, we need to handle chunks of blocks like we did with threads in normal programming.

Analysis quiz 7:

sequential: O(N) + O(N) = cost and latency
parallel: work: O(nlogn) because 2 need 2 * log(n) ops for bsearch, n indexes work on the merge, each calls 2 * log(n) => total is O(n * log(n)); latency: 2 * log(n) because all threads running in parallel, only take 2 * log(n) time for binary search 2 times

Read a certain number of lines in a file: `head -n4 x_y.csv > x_y_3.csv`