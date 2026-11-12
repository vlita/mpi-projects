#include "pi.h"

double pi_calc(long int n, int rank, int size){

    // handle n/p != integer
    long int npoints = (n - (n % size)) / size;

    if (rank == 0) {
        npoints += n % size;
    }

    // initialize sum to be communicated
    long int sum = 0;

    // seed random generator
    std::srand(std::time(NULL) + rank);
    // make a change to this file
    // generate points and add to sum if inside circle
    for (long int i = 0; i < npoints; ++i) {
        double x = static_cast<double>(std::rand()) / RAND_MAX;
        double y = static_cast<double>(std::rand()) / RAND_MAX;
        x *= x;
        y *= y;

        if ((x + y) < 1.0) {
            sum += 1;
        }
    }

    // perform ceil(log_2(p)) w/o cmath dependency
    int d = 0;
    if(size != 1) {
        int x = size - 1;
        while ((x) >>= 1) d++;
        d++;
    }
    
    // hypercubic permutation 
    for (int i = 0; i < d; ++i) {
        long int received = 0;

        if ((rank & (1 << i)) != 0) {
            MPI_Send(&sum, 1, MPI_LONG, 
                     (rank ^ (1 << i)), 1, MPI_COMM_WORLD);
            break;

        } else {
            if ((rank ^ (1 << i)) < size) {
                MPI_Status stat; 

                MPI_Recv(&received, 1, MPI_LONG,
                        (rank ^ (1 << i)), MPI_ANY_TAG, MPI_COMM_WORLD, 
                        &stat);
                sum += received;

            } else {
                continue;
                
            }
        }
    }

    // estimate pi
    double result = 0;
    if (rank == 0) {
        result = 4 * static_cast<double>(sum) / n;
    }

    return result;
}
