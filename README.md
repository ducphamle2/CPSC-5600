# CPSC-5600
Parallel Computing course Seattle University


bitonic sort, comparators, network diagram, code similar to what we have done in the hw
also has synchronous queue question
analysis

cost: how much CPU time did it cost in total - how many clock cycles im using - how many basic operations we need to run
for example: we need two for loop of n in bubble sort => total operations is worst at n^2

cost analysis steps (Need to remember the steps when working on the exam):
1. size of the input (eg: N). Need to tell what parameters contribute to the total size of the input in the exam. Otherwise has no full credits. Parameters should vary or change, not constant. If a variable is constant => no need to analyze it.
2. basic ops - total number of basic ops

3. Is cost dependent of the quality of the input? Oblivious algorithms cost is independent

4. recursive relation
5. solve recurrence

for 4 & 5, only need to count something from the behaviour of the structure of the thing we are doing the recursive. For example, complete binary tree has h = log2(n+1), and binary tree is log2(n)

latency: wall clock - total elapsed time

use data flow analysis. Consider a starting point and stop point. Form a slowest path from the start to stop => the path is the latency.

big O can be: O(n / p) where p is the number of processes, n is the size