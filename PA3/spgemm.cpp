#include <vector>
#include <map>
#include <algorithm>
#include <utility>
#include <iostream>
#include <mpi.h>
#include <cassert>
#include <cmath>
#include "functions.h"

void spgemm_2d(int m, int p, int n,
               std::vector<std::pair<std::pair<int,int>, int>> &A,
               std::vector<std::pair<std::pair<int,int>, int>> &B,
               std::vector<std::pair<std::pair<int,int>, int>> &C,
               std::function<int(int, int)> plus, std::function<int(int, int)> times,
               MPI_Comm row_comm, MPI_Comm col_comm)
{
    // retrieve WORLD-level info
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int q = sqrt(size);

    // row coordinate and column coordinate
    int row_rank, col_rank;
    MPI_Comm_rank(row_comm, &row_rank);
    MPI_Comm_rank(col_comm, &col_rank);

    int row_size, col_size;
    MPI_Comm_size(row_comm, &row_size);
    MPI_Comm_size(col_comm, &col_size);


    // calculate bounds on p index block (cols of A, rows of B)
    std::vector<int> t_displs(q);
    std::vector<int> t_counts(q);

    int t_p = p % q;
    int t_block = (p - t_p) / q;

    int sum_p = 0;
    for (int i = 0; i < q; i++) {
        t_counts[i] = t_block;
        if (t_p-- > 0) t_counts[i]++;
        t_displs[i] = sum_p;
        sum_p += t_counts[i];
    }

    int A_p_start = t_displs[row_rank];
    int A_p_count = t_counts[row_rank];
    int B_p_start = t_displs[col_rank];
    int B_p_count = t_counts[col_rank];

    // convert A: COO -> CSC over local columns
    std::vector<int> A_col_ptr(A_p_count + 1, 0);
    std::vector<int> A_row_idx(A.size());
    std::vector<int> A_val(A.size());

    // calculate column locations
    for (const auto& e : A) {
        int c_local = e.first.second - A_p_start;
        if (c_local >= 0 && c_local < A_p_count) {
            A_col_ptr[c_local + 1]++;
        }
    }

    // offset by idx
    for (int c = 0; c < A_p_count; ++c) {
        A_col_ptr[c + 1] += A_col_ptr[c];
    }

    // build row idx and val arrays
    std::vector<int> A_next = A_col_ptr;
    for (const auto& e : A) {
        int c_local = e.first.second - A_p_start;
        if (c_local < 0 || c_local >= A_p_count) {
            continue;
        }
        int r = e.first.first;
        int v = e.second;
        int pos = A_next[c_local]++;
        A_row_idx[pos] = r;
        A_val[pos] = v;
    }

    // convert B: COO -> CSR over local rows
    std::vector<int> B_row_ptr(B_p_count + 1, 0);
    std::vector<int> B_col_idx(B.size());  // can do this because Bij <- (Bij)^T called before spgemm
    std::vector<int> B_val(B.size());

    // calculate row locations
    for (const auto& e : B) {
        int r_local = e.first.first - B_p_start;
        if (r_local >= 0 && r_local < B_p_count) {
            B_row_ptr[r_local + 1]++;
        }
    }

    // offset by idx
    for (int r = 0; r < B_p_count; ++r) {
        B_row_ptr[r + 1] += B_row_ptr[r];
    }

    // build col idx and val arrays
    std::vector<int> B_next = B_row_ptr;
    for (const auto& e : B) {
        int r_local = e.first.first - B_p_start;
        if (r_local < 0 || r_local >= B_p_count) {
            continue;
        }
        int c = e.first.second;
        int v = e.second;
        int pos = B_next[r_local]++;
        B_col_idx[pos] = c;
        B_val[pos] = v;
    }

    // SUMMA main loop
    std::map<std::pair<int,int>, int> C_accum; // hashmap to accumulate final values into 
    for (int t = 0; t < q; t++) {

        // slice of A local columns
        int AB_width = t_counts[t];
        int A_ptr_length = AB_width + 1;
        int A_nonzero = 0;

        // broadcast nonzero elements in bounds from CSC 
        std::vector<int> A_ptr_slice; 
        std::vector<int> A_row_slice;
        std::vector<int> A_val_slice;

        // resize slices
        if (row_rank == t) { 
            int start = A_col_ptr[0];
            int end   = A_col_ptr[AB_width];
            A_nonzero = end - start;
            A_ptr_slice.resize(A_ptr_length);

            // calculate size of slice over col pointers
            for (int x = 0; x < A_ptr_length; x++) {
                A_ptr_slice[x] = A_col_ptr[x] - start;
            }

            A_row_slice.assign(A_row_idx.begin() + start, A_row_idx.begin() + end);
            A_val_slice.assign(A_val.begin() + start, A_val.begin() + end);
        }

        // resize buffers by # nonzero elements
        MPI_Bcast(&A_nonzero, 1, MPI_INT, t, row_comm);

        if (row_rank != t) {
            A_ptr_slice.resize(A_ptr_length);
            A_row_slice.resize(A_nonzero);
            A_val_slice.resize(A_nonzero);
        }

        MPI_Bcast(A_ptr_slice.data(), A_ptr_length, MPI_INT, t, row_comm);
        MPI_Bcast(A_row_slice.data(), A_nonzero, MPI_INT, t, row_comm);
        MPI_Bcast(A_val_slice.data(), A_nonzero, MPI_INT, t, row_comm);

        // slice of B local rows
        int B_ptr_length = AB_width + 1;
        int B_nonzero = 0;

        // broadcast nonzero elements in bounds from CSR
        std::vector<int> B_ptr_slice;
        std::vector<int> B_col_slice;
        std::vector<int> B_val_slice;

        // resize slices
        if (col_rank == t) {
            int start = B_row_ptr[0];
            int end   = B_row_ptr[AB_width];
            B_nonzero = end - start;
            B_ptr_slice.resize(B_ptr_length);

            // calculate size of slice over row pointers
            for (int x = 0; x < B_ptr_length; x++) {
                B_ptr_slice[x] = B_row_ptr[x] - start;
            }

            B_col_slice.assign(B_col_idx.begin() + start, B_col_idx.begin() + end);
            B_val_slice.assign(B_val.begin() + start, B_val.begin() + end);
        }

        // resize buffers by # nonzero elements
        MPI_Bcast(&B_nonzero, 1, MPI_INT, t, col_comm);

        if (col_rank != t) {
            B_ptr_slice.resize(B_ptr_length);
            B_col_slice.resize(B_nonzero);
            B_val_slice.resize(B_nonzero);
        }

        MPI_Bcast(B_ptr_slice.data(), B_ptr_length, MPI_INT, t, col_comm);
        MPI_Bcast(B_col_slice.data(), B_nonzero, MPI_INT, t, col_comm);
        MPI_Bcast(B_val_slice.data(), B_nonzero, MPI_INT, t, col_comm);

        // CSC for A and CSR for B matches contraction pattern A(i,k)*B(k,j)
        for (int k = 0; k < AB_width; k++) { // A_ptr_slice and B_ptr_slice are the same size (per CS format)
            int A_begin = A_ptr_slice[k];
            int A_end   = A_ptr_slice[k + 1];
            int B_begin = B_ptr_slice[k];
            int B_end   = B_ptr_slice[k + 1];

            // automatically filters zero-valued columns (per CS)
            for (int a = A_begin; a < A_end; a++) {
                int i = A_row_slice[a];
                int A_val = A_val_slice[a];

                // automatically filters zero-valued rows (per CS)
                for (int b = B_begin; b < B_end; b++) {
                    int j = B_col_slice[b];
                    int B_val = B_val_slice[b];
                    int prod = times(A_val, B_val);
                    auto key = std::make_pair(i, j);
                    auto it = C_accum.find(key);

                    // accumulate into hashmap!
                    if (it == C_accum.end()) {
                        C_accum[key] = prod;
                    } else {
                        it->second = plus(it->second, prod);
                    }
                }
            }        
        }
    }

    // add to local C matrix
    for (const auto& kv : C_accum) {
        C.push_back({{kv.first.first, kv.first.second}, kv.second});
    }
}
