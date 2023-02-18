/**
 * CPSC 5600, Quiz 4, Question 2
 * Seattle U, Winter 2023
 * Kevin Lundeen
 */

#include <iostream>
#include <vector>
#include <array>
#include "mpi.h"

using namespace std;

class Quiz4Question2 {
public:
    /**
     * Set up an instance with the data shown on the quiz example (plus a 7th bucket)
     * then set up the preconditions for scatterBuckets, call it, and then print out the
     * m and partition which are promised by scatterBuckets postconditions.
     */
    static void test() {
        Quiz4Question2 t;
        MPI_Comm_size(MPI_COMM_WORLD, &t.p);
        MPI_Comm_rank(MPI_COMM_WORLD, &t.rank);
        if (t.rank == t.ROOT) {
            t.n = 7;
            t.table = new vector<array<u_short, 2>>[t.n];
            t.table[0].push_back({16, 1301});
            t.table[0].push_back({1, 919});
            t.table[1].push_back({0, 200});
            t.table[1].push_back({6, 3});
            t.table[1].push_back({14, 999});
            t.table[3].push_back({166, 9098});
            t.table[3].push_back({666, 634});
            t.table[3].push_back({876, 12});
            t.table[3].push_back({6, 7});
            t.table[4].push_back({10, 923});
            t.table[5].push_back({11, 4});
            t.table[5].push_back({12, 5});
            t.table[6].push_back({920, 91}); // can't remember what we had on the board...
            t.table[6].push_back({1409, 500});
        }

        // send n to everyone
        MPI_Bcast(&t.n, 1, MPI_INT, t.ROOT, MPI_COMM_WORLD);

        // test scatterBuckets
        t.scatterBuckets();
        cout << t.rank << " has partition (size: " << t.m << "): ";
        for (int bi = 0; bi < t.m; bi++) {
            cout << "{";
            for (const auto elem : t.partition[bi])
                cout << "<" << elem[0] << "," << elem[1] << ">";
            cout << "}, ";
        }
        cout << endl;
    }

private:
    int n = 0;
    vector<array<u_short,2>> *table = nullptr;
    int p = 1;
    const int ROOT=0;
    int rank = ROOT;
    const int MAX_BUCKET_SIZE=5;
    vector<array<u_short,2>> *partition = nullptr;
    int m = 0; // size of partition


    /**
     * @pre n, p are set for all; table is set for ROOT only
     * @post m, partition are set
     */
    void scatterBuckets() {
        u_short *sendbuf = nullptr, *recvbuf = nullptr;  // nullptr allows delete to work for anyone
        int *sendcounts = nullptr, *displs = nullptr;
        int buckets_per_proc = n / p;

        if (rank == ROOT) {
            // marshal data into sendbuf and set up sending side of message (ROOT only)
            sendbuf = new u_short[n * (1 + 2 * MAX_BUCKET_SIZE)]; // max size
            sendcounts = new int[p];
            displs = new int[p];
            int i = 0;  // index into sendbuf
            for (int pi = 0; pi < p; pi++) {
                displs[pi] = i;
                int begin_bucket = buckets_per_proc * pi;
                int end_bucket = begin_bucket + buckets_per_proc;
                if (pi == p - 1)
                    end_bucket += n - buckets_per_proc * p; // extras for last proc
                for (int bi = begin_bucket; bi < end_bucket; bi++) {
                    sendbuf[i++] = table[bi].size();
                    for (const auto hash_entry : table[bi]) {
                        sendbuf[i++] = hash_entry[0];
                        sendbuf[i++] = hash_entry[1];
                    }
                }
                sendcounts[pi] = i - displs[pi];
            }
        }

        // set this->m for my process
        m = buckets_per_proc;
        if (rank == p - 1)
            m += n - buckets_per_proc * p;

        // set up receiving side of message (everyone)
        int recvcount = m * (1 + 2 * MAX_BUCKET_SIZE);
        recvbuf = new u_short[recvcount];

        MPI_Scatterv(sendbuf, sendcounts, displs, MPI_UNSIGNED_SHORT,
                     recvbuf, recvcount, MPI_UNSIGNED_SHORT,
                     ROOT, MPI_COMM_WORLD);

        // unmarshal data from recvbuf into this->partion
        partition = new vector<array<u_short,2>>[m];  // calls default ctor for each
        int j = 0; // index into recvbuf
        for (int bi = 0; bi < m; bi++) {
            int esize = recvbuf[j++];
            for (int ei = 0; ei < esize; ei++ ) {
                u_short key = recvbuf[j++];
                u_short value = recvbuf[j++];
                array<u_short,2> entry = {key, value};
                partition[bi].push_back(entry);
            }
        }

        // free temp arrays
        delete[] sendbuf;
        delete[] sendcounts;
        delete[] displs;
        delete[] recvbuf;
    }
};

int main() {
    MPI_Init(nullptr, nullptr);
    Quiz4Question2::test();
    MPI_Finalize();
}
