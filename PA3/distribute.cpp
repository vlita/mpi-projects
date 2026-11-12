#include <algorithm>
#include <vector>
#include <iostream>
#include <fstream>
#include <cassert>
#include <cmath>
#include <string>
#include <sstream>
#include <numeric>
#include "functions.h"

void distribute_matrix_2d(int m, int n, std::vector<std::pair<std::pair<int, int>, int>> &full_matrix,
                          std::vector<std::pair<std::pair<int, int>, int>> &local_matrix,
                          int root, MPI_Comm comm_2d)
{
    // retrieve WORLD-level info
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int q = sqrt(size);

    // set up row and column communicators
    int dims[2], periods[2], coords[2];
    MPI_Cart_get(comm_2d, 2, dims, periods, coords);

    // using Cart_sub instead of Comm_split
    int remain_row[2] = {0, 1};
    int remain_col[2] = {1, 0};
    MPI_Comm row_comm_d, col_comm_d;
    MPI_Cart_sub(comm_2d, remain_row, &row_comm_d);
    MPI_Cart_sub(comm_2d, remain_col, &col_comm_d);

    // calculate root from comm_2d (should be the same as WORLD root)
    int root_coords[2] = {0, 0};
    int cart_root;
    MPI_Cart_rank(comm_2d, root_coords, &cart_root);

    // calculate root process along each row
    int row_root_coords[1] = {0};
    int row_root;
    MPI_Cart_rank(row_comm_d, row_root_coords, &row_root);

    // calculate root process along each column
    int col_root_coords[1] = {0};
    int col_root;
    MPI_Cart_rank(col_comm_d, col_root_coords, &col_root);

    // calculate bounds on row & column indicies (for blocking)
    int r_m = m % q;
    int m_block = (m - r_m) / q;

    int r_n = n % q;
    int n_block = (n - r_n) / q;

    std::vector<int> displs_m(q);
    std::vector<int> sendcounts_m(q);

    std::vector<int> displs_n(q);
    std::vector<int> sendcounts_n(q);

    int sum_m = 0;
    int sum_n = 0;
    for (int i=0; i<q; i++) {
        sendcounts_m[i] = m_block;
        sendcounts_n[i] = n_block;
        if (r_m-- > 0) sendcounts_m[i]++;
        if (r_n-- > 0) sendcounts_n[i]++;
        displs_m[i] = sum_m;
        sum_m += sendcounts_m[i];
        displs_n[i] = sum_n;
        sum_n += sendcounts_n[i];
    }

    // calculate elements to be recieved by first column of processors in cart 
    std::vector<int> first_sendcounts(q);
    std::vector<int> first_displs(q);
    std::vector<int> first_row_idx_sendbuf;
    std::vector<int> first_col_idx_sendbuf;
    std::vector<int> first_vals_sendbuf;

    if(coords[0] == 0 && coords[1] == 0) {
        int r = 0;
        for (const auto& e : full_matrix) {
            int row = e.first.first;
            while (row >= displs_m[r] + sendcounts_m[r]) r++;
            first_sendcounts[r]++;
        }

        first_displs[0] = 0;
        for (int i = 1; i < q; i++) {
            first_displs[i] = first_displs[i-1] + first_sendcounts[i-1];
        }
        
        first_row_idx_sendbuf.reserve(full_matrix.size());  // need to release as well
        first_col_idx_sendbuf.reserve(full_matrix.size());
        first_vals_sendbuf.reserve(full_matrix.size());
        
        for (const auto& e : full_matrix) {
            first_row_idx_sendbuf.push_back(e.first.first);
            first_col_idx_sendbuf.push_back(e.first.second);
            first_vals_sendbuf.push_back(e.second);
        }
    }

    // see footnote 
    
    std::vector<int> second_sendcounts(q);
    std::vector<int> second_displs(q);
    std::vector<int> second_row_idx_sendbuf;
    std::vector<int> second_col_idx_sendbuf;
    std::vector<int> second_vals_sendbuf;
    
    if(coords[1] == 0) {
        int first_sendcount_recv;

        MPI_Scatter(first_sendcounts.data(), 1, MPI_INT, 
                    &first_sendcount_recv, 1, MPI_INT, cart_root, col_comm_d);

        std::vector<int> first_row_idx_recvbuf(first_sendcount_recv);
        std::vector<int> first_col_idx_recvbuf(first_sendcount_recv);
        std::vector<int> first_vals_recvbuf(first_sendcount_recv);

        MPI_Scatterv(first_row_idx_sendbuf.data(), first_sendcounts.data(), first_displs.data(), MPI_INT,
                     first_row_idx_recvbuf.data(), first_sendcount_recv, MPI_INT,
                     cart_root, col_comm_d);
        
        MPI_Scatterv(first_col_idx_sendbuf.data(), first_sendcounts.data(), first_displs.data(), MPI_INT,
                     first_col_idx_recvbuf.data(), first_sendcount_recv, MPI_INT,
                     cart_root, col_comm_d);

        MPI_Scatterv(first_vals_sendbuf.data(), first_sendcounts.data(), first_displs.data(), MPI_INT,
                     first_vals_recvbuf.data(), first_sendcount_recv, MPI_INT,
                     cart_root, col_comm_d);

        // free up large arrays since we no longer need them
        if(coords[0] == 0 && coords[1] == 0) {
            first_row_idx_sendbuf.clear();
            first_col_idx_sendbuf.clear();
            first_vals_sendbuf.clear();
        }

        // need to sort buffers by col idx so that bucketing is correct
        std::vector<int> permute_col(first_sendcount_recv);
        std::iota(permute_col.begin(), permute_col.end(), 0);

        // sort indices based on first_col_idx_recvbuf
        std::sort(permute_col.begin(), permute_col.end(),
            [&](int a, int b) {
                return first_col_idx_recvbuf[a] < first_col_idx_recvbuf[b];
            });

        std::vector<int> sorted_c(first_sendcount_recv);
        std::vector<int> sorted_r(first_sendcount_recv);
        std::vector<int> sorted_v(first_sendcount_recv);

        for (int i = 0; i < first_sendcount_recv; i++) {
            sorted_c[i] = first_col_idx_recvbuf[permute_col[i]];
            sorted_r[i] = first_row_idx_recvbuf[permute_col[i]];
            sorted_v[i] = first_vals_recvbuf[permute_col[i]];
        }

        first_col_idx_recvbuf = std::move(sorted_c);
        first_row_idx_recvbuf = std::move(sorted_r);
        first_vals_recvbuf = std::move(sorted_v);

        // calculate # of elements to be recieved by each row of processors in row_comm
        int c = 0;
        for (int i = 0; i < first_sendcount_recv; i++) {
            int col = first_col_idx_recvbuf[i];
            while (col >= displs_n[c] + sendcounts_n[c]) c++;
            second_sendcounts[c]++;
        }

        second_displs[0] = 0;
        for (int i = 1; i < q; i++) {
            second_displs[i] = second_displs[i-1] + second_sendcounts[i-1];
        }
        
        second_row_idx_sendbuf.reserve(first_sendcount_recv); // probably a lot smaller... do I need to clear?
        second_col_idx_sendbuf.reserve(first_sendcount_recv);
        second_vals_sendbuf.reserve(first_sendcount_recv); 
        
        // bucket sorted column data according to second_sendcount bounds
        for (int i = 0; i < first_sendcount_recv; i++) {
            second_row_idx_sendbuf.push_back(first_row_idx_recvbuf[i]);
            second_col_idx_sendbuf.push_back(first_col_idx_recvbuf[i]);
            second_vals_sendbuf.push_back(first_vals_recvbuf[i]);
        }
    }

    int second_sendcount_recv;

    MPI_Scatter(second_sendcounts.data(), 1, MPI_INT, 
                &second_sendcount_recv, 1, MPI_INT, row_root, row_comm_d);

    std::vector<int> second_row_idx_recvbuf(second_sendcount_recv);
    std::vector<int> second_col_idx_recvbuf(second_sendcount_recv);
    std::vector<int> second_vals_recvbuf(second_sendcount_recv);

    MPI_Scatterv(second_row_idx_sendbuf.data(), second_sendcounts.data(), second_displs.data(), MPI_INT,
                 second_row_idx_recvbuf.data(), second_sendcount_recv, MPI_INT,
                 row_root, row_comm_d);
    
    MPI_Scatterv(second_col_idx_sendbuf.data(), second_sendcounts.data(), second_displs.data(), MPI_INT,
                 second_col_idx_recvbuf.data(), second_sendcount_recv, MPI_INT,
                 row_root, row_comm_d);

    MPI_Scatterv(second_vals_sendbuf.data(), second_sendcounts.data(), second_displs.data(), MPI_INT,
                 second_vals_recvbuf.data(), second_sendcount_recv, MPI_INT,
                 row_root, row_comm_d);

    MPI_Comm_free(&row_comm_d);
    MPI_Comm_free(&col_comm_d);

    for (int i = 0; i < second_sendcount_recv; ++i) {
        local_matrix.push_back(
            {{second_row_idx_recvbuf[i], second_col_idx_recvbuf[i]}, 
              second_vals_recvbuf[i]});
    }

}


// footnote
// At this point each processor in the grid knows which columns and which rows of the original matrix it is assigned, as well as the range of indicies (i,j) it will recieve
// the root rank knows the index ranges (and associated entries) of the original matrix (i,j) that lies within the row range assigned to each proccesor along the cartesian y-axis (dim 1). 
// The arrays storing this information must be communicated to the "first column" of the cartesian communicator (that is, all processors in the range (0,0) to (sqrt(p),0))
// To do this, we need to set up a custom communicator only involving the first column of processors, and scatter both the count array and the displacements array.
// Now we need to perform 3 individual scatterv operations of chunks of 3 arrays initialized on the root processor (row_idx, col_idx, vals) and recieve into buffers specified by count and and displacements array recieved in the previous step.
// Now we release the custom communicator, and on each procesor in the range (0,0) to (sqrt(p),0) in parallel we perform the same computations as above. 
// That is, we calculate the entry range and associated indices that must be passed to each processor among each row of processors (dim 0) in the cartesian communicator (the logic is the same modulo communication patter). 
