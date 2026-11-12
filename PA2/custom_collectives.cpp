#include "custom_collectives.h"
#include <algorithm> 
#include <vector>
#include <mpi.h>

void Custom_Scatter(int* sendbuf, int sendcount, MPI_Datatype sendtype,
                    int* recvbuf, int recvcount, MPI_Datatype recvtype,
                    int root, MPI_Comm comm) {

    // setup
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    // perform d = ceil(log_2(p)) w/o cmath dependency
    int d = 0;
    if(size != 1) {
        int x = size - 1;
        while ((x) >>= 1) d++;
        d++;
    }

    // calculate message length 
    int message_length = sendcount * size;

    // calculate virtual rank
    int virt = ((rank - root) + size) % size;

    // shift sendbuf by root rank 
    if (rank == root) {
        std::rotate(sendbuf, sendbuf + (root * sendcount), sendbuf + (message_length));
    }
    
    // scatter with hypercubic permutation
    int flip = 1 << (d-1);
    int mask = flip - 1;
    int recv_idx = sendcount * virt;
    
    for(int j = d; j > 0; j--) {
        message_length = message_length / 2;

        if((virt & mask) == 0) {
            int send_idx = sendcount * (virt ^ flip);

            if((virt & flip) == 0) {
                MPI_Send(&sendbuf[send_idx], message_length, sendtype, 
                        (((virt ^ flip) + root) % size), 11, comm);
            } else {
                MPI_Status stat;
                MPI_Recv(&sendbuf[recv_idx], message_length, recvtype,
                        (((virt ^ flip) + root) % size), MPI_ANY_TAG, comm, &stat);
            }
        }
        mask = mask >> 1;
        flip = flip >> 1;
    }

    std::copy(&sendbuf[recv_idx], &sendbuf[recv_idx + recvcount], recvbuf);
}

void Custom_Allgather(int* sendbuf, int sendcount, MPI_Datatype sendtype,
                      int* recvbuf, int recvcount, MPI_Datatype recvtype,
                      MPI_Comm comm) {
    // Write your code below
    ////////////////////////////////////////
    int p, rank;
    MPI_Comm_size(comm, &p);
    MPI_Comm_rank(comm, &rank);

    // Place local chunk into its final slot
    for (int i = 0; i < recvcount; ++i) recvbuf[rank * recvcount + i] = sendbuf[i];

    int rounds = 0;
    while ((1 << rounds) < p) ++rounds;

    int group_size = 1; // measured in chunks (ranks)
    for (int k = 0; k < rounds; ++k) {
        int partner = rank ^ (1 << k);

        int block_start         = (rank    / group_size) * group_size;
        int partner_block_start = (partner / group_size) * group_size;

        int send_offset = block_start * recvcount;
        int recv_offset = partner_block_start * recvcount;
        int elems       = group_size * recvcount;

        MPI_Sendrecv(recvbuf + send_offset, elems, MPI_INT, partner, k,
                     recvbuf + recv_offset, elems, MPI_INT, partner, k,
                     comm, MPI_STATUS_IGNORE);

        group_size <<= 1;
    }
    ////////////////////////////////////////
}

void Custom_Allreduce(int* sendbuf, int* recvbuf, int count,
                      MPI_Datatype datatype, MPI_Op op, MPI_Comm comm) {
    // Write your code below
    ////////////////////////////////////////
    int p, rank;
    MPI_Comm_size(comm, &p);
    MPI_Comm_rank(comm, &rank);

    for (int i = 0; i < count; ++i) recvbuf[i] = sendbuf[i];

    int rounds = 0;
    while ((1 << rounds) < p) ++rounds;

    std::vector<int> tmp(count);
    // Hypercube recursive doubling for SUM
    for (int k = 0; k < rounds; ++k) {
        int partner = rank ^ (1 << k);
        MPI_Sendrecv(recvbuf, count, datatype, partner, k,
                     tmp.data(), count, datatype, partner, k,
                     comm, MPI_STATUS_IGNORE);
        for (int i = 0; i < count; ++i) recvbuf[i] += tmp[i];
    }
    ////////////////////////////////////////
}

void Custom_Alltoall_Hypercube(int* sendbuf, int sendcount, MPI_Datatype sendtype,
                               int* recvbuf, int recvcount, MPI_Datatype recvtype,
                               MPI_Comm comm) {

    // setup
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    // ceil(log_2(p))
    int d = 0;
    while ((1 << d) < size) ++d;

    //calculate message length
    int message_length = (sendcount * size) / 2;

    // //temporary buffer
    std::vector<int> tempbuf(sendcount * size);

    // all to all with hypercubic permutation
    int flip = 1 << (d-1);
    for (int k = 0; k < d; ++k) {
        
        // if partner's rank > own rank, right half of data is sent, else left half is sent
        int idx = ((rank ^ flip) > rank) ? message_length : 0;
        MPI_Sendrecv(&sendbuf[idx], message_length, sendtype,
                     (rank ^ flip), 11, recvbuf, message_length, recvtype,
                     (rank ^ flip), MPI_ANY_TAG, comm, MPI_STATUS_IGNORE);

        // interleave chunks based on partner rank 
        for (int i = 0; i < message_length; i += sendcount) {
            if (idx == 0) {
                std::copy(&recvbuf[i], &recvbuf[i + sendcount], &tempbuf[2*i]);
                std::copy(&sendbuf[message_length + i], &sendbuf[message_length + i + sendcount], 
                          &tempbuf[2*i + sendcount]);

            } else {
                std::copy(&sendbuf[i], &sendbuf[i + sendcount], &tempbuf[2*i]);
                std::copy(&recvbuf[i], &recvbuf[i + sendcount], &tempbuf[2*i + sendcount]);
            }
        }
        
        std::copy(tempbuf.begin(), tempbuf.end(), sendbuf);
        flip = flip >> 1;
    }
    
    std::copy(sendbuf, sendbuf + (sendcount * size), recvbuf);

    // for (int k = 0; k < (sendcount * size); ++k) {
    //     std::cout << recvbuf[k] << " ";
    //     std::cout << std::endl;
    // }
}

void Custom_Alltoall_Arbitrary(int* sendbuf, int sendcount, MPI_Datatype sendtype,
                               int* recvbuf, int recvcount, MPI_Datatype recvtype,
                               MPI_Comm comm) {

    // setup
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    // copy "diagonal" data into recvbuf
    std::copy(&sendbuf[rank * sendcount], &sendbuf[(rank * sendcount) + sendcount], &recvbuf[rank * recvcount]);

    // all to all with arbitrary permutation
    for(int j = 1; j < size; j++){
        int send_idx = (rank + j) % size; // always sending to projessor j ranks ahead
        int recv_idx = ((rank - j) + size) % size; // always receiving from projessor j ranks behind

        MPI_Status stat;
        MPI_Sendrecv(&sendbuf[sendcount * send_idx], sendcount, sendtype,
                     send_idx, 11,
                     &recvbuf[sendcount * recv_idx], recvcount, recvtype,
                     recv_idx, MPI_ANY_TAG, comm, &stat);
    }
}
