# Notes when using Java for the homework and quizzes

## R3 - Java Concurrency

- In Java, concurrent prog = threads.
- Time slicing: Processing time for a core share among processes & threads.
- Concurrency possible with 1 core.
- Process:
    - Most JVMs are implemented to run in 1 process, where each process has at least 1 thread
    - Process is a self-contained execution env. Has its own memory space & run-time resources.
- Thread:
    - Each Java program starts at least a thread called main thread. The main thread can create additional threads.
    - Threads are lightweight processes, share memory, open files and other run-time resources with each other.
    - Threads are within a process.

Problems with threads in Java that require synchronization:
- Interleaved ops - ops are overlapped when they shouldnt.
- Inconsistent data aka memory inconsistent - when thread B reads before thread A writes to the same data.
