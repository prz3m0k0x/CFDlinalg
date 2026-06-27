#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "fvm.h"


fvm_csm_mat* fvm_csm_mat_creator(unsigned int n_rows, unsigned int n_cols, unsigned int n_nonzero)
{
    if (n_rows == 0) {
        CFD_ERROR(CFD_INVALID_DIMENSIONS);
        return NULL;
    }

    fvm_csm_mat* res = malloc(sizeof *res);
    if (res == NULL) return NULL;
    res->num_rows = n_rows;
    res->num_cols = n_cols;
    res->n_nonzero = n_nonzero;
    res->data = malloc(sizeof(res->data[0]) * n_nonzero);
    res->col_ind = malloc(sizeof(res->col_ind[0]) * n_nonzero);
    res->indptr = malloc(sizeof(res->indptr[0]) * (size_t)(n_rows + 1));

    if (res->data == NULL || res->col_ind == NULL || res->indptr == NULL) {
        free(res->data);
        free(res->col_ind);
        free(res->indptr);
        free(res);
        return NULL;
    }

    return res;
}

void fvm_csm_mat_destroyer(fvm_csm_mat* csm) {
  if (csm == NULL) return;

  free(csm->data);
  free(csm->col_ind);
  free(csm->indptr);
  free(csm);
}

fvm_csm_mat* fvm_csm_from_mat(fvm_mat *m)
{
    if (m == NULL) {
        CFD_ERROR(CFD_NULL_POINTER);
        return NULL;
    }

    size_t n_nonzero = fvm_mat_nonzero_num(m, 1e-10);
    fvm_csm_mat* csm = fvm_csm_mat_creator(m->num_rows, m->num_cols, n_nonzero);
    if (csm == NULL) return NULL;

    size_t count = 0;
    csm->indptr[0] = 0;

    for (size_t i = 0; i < m->num_rows; i++) {
        for (size_t j = 0; j < m->num_cols; j++) {
            double val = MAT_AT(m, i, j);
            if (fabs(val) > 1e-10) {
                csm->data[count] = val;
                csm->col_ind[count] = j;
                count++;
            }
        }
        csm->indptr[i + 1] = count;
    }

    return csm;
}
fvm_csm_mat* fvm_csm_transpose(fvm_csm_mat* csm)
{
    if (csm == NULL) {
        CFD_ERROR(CFD_NULL_POINTER);
        return NULL;
    }

    fvm_csm_mat* res = fvm_csm_mat_creator(csm->num_cols, csm->num_rows, csm->n_nonzero);
    if (res == NULL) {CFD_ERROR(CFD_NULL_POINTER); return NULL;}

    size_t ncols = csm->num_cols;

    size_t* counts = calloc(ncols, sizeof(*counts));
    if (counts == NULL) {
        fvm_csm_mat_destroyer(res);
        return NULL;
    }

    for (size_t i = 0; i < csm->n_nonzero; i++) {
        counts[csm->col_ind[i]]++;
    }

    res->indptr[0] = 0;
    for (size_t i = 0; i < ncols; i++) {
        res->indptr[i + 1] = res->indptr[i] + counts[i];
    }

    size_t* next = malloc(ncols * sizeof(*next));
    if (next == NULL) {
        free(counts);
        fvm_csm_mat_destroyer(res);
        return NULL;
    }

    for (size_t i = 0; i < ncols; i++) {
        next[i] = res->indptr[i];
    }

    for (size_t row = 0; row < csm->num_rows; row++) {
        for (size_t j = csm->indptr[row]; j < csm->indptr[row + 1]; j++) {
            size_t col = csm->col_ind[j];
            size_t dest = next[col]++;

            res->data[dest] = csm->data[j];
            res->col_ind[dest] = row;
        }
    }

    free(counts);
    free(next);
    return res;
}