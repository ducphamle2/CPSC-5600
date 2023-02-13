all processes run the exact same code, different in rank. Each process has a unique rank.

Can run different source codes by if else check the rank in the code, but same executables

MPI_Send doenst block. Once the MPI routine successfully collects & store the data passed in the MPI_Send into the system buffer, MPI will unblock. If the system buffer cannot store the data => program gets blocked => be careful, if the data grows larger, the program may be blocked

SSend always block the program until Recv gets the message

MPI_Scatter - array chunks received based on the process rank. Rank 0 gets first chunk, rank 1 gets 2nd chunk,... Cannot change the chunk we want to receive.

pick 10 points random in dataset. Each data item => which point are you close to => stick to that cluster. What's your average? What's your center? How far an element from the center => if find something closer => move again until everybody is in their correct clusters. Question: How do we know what cluster should an element be in?