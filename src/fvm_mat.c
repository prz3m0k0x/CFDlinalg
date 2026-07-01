#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "../include/CFDlinalg/fvm.h"
#include <omp.h>
#include <math.h>
#include <limits.h>
#include <time.h>
/*
Functions suffixed with _r return the augmented object, e.g. original matrix mutliplied by scalar,
while functions without such suffix create a copy of an object. Care is advised when using creator
functions, as memory management might become an issue. */

/*
Functions suffixed with _mp signals that function is intended to be used in parallel computing. 
Such functions are under construction as of right now and it is advised not to use them.
*/

/* --- Creator, destroyer, checkers, basic utilities --- */
fvm_mat *fvm_mat_creator(unsigned int num_rows, unsigned int num_cols);
fvm_mat *fvm_mat_decl(const double *arr, unsigned int num_rows, unsigned int num_cols);
void fvm_mat_destroyer(fvm_mat *m);
int fvm_mat_nonzero_count(fvm_mat *m, double tol);
fvm_mat *fvm_mat_rand(unsigned int num_rows, unsigned int num_cols, double min, double max);
fvm_mat *fvm_mat_square(unsigned int num);
fvm_mat *fvm_mat_eye(unsigned int num);
fvm_mat *fvm_mat_ones(unsigned int num_rows, unsigned int num_cols);
fvm_mat *fvm_mat_full(unsigned int num_rows, unsigned int num_cols, double a);
fvm_mat *fvm_mat_diag(unsigned int num_rows, unsigned int num_cols, double val);
fvm_mat *fvm_mat_copy(fvm_mat *m);

/* --- Shape, size and matrix alternatiion and manipulation functions --- */
fvm_mat* fvm_mat_remove_row(fvm_mat* m, unsigned int row_n);
fvm_mat* fvm_mat_remove_col(fvm_mat* m, unsigned int col_n);
int fvm_mat_swap_rows_r(fvm_mat* m, unsigned int r1, unsigned int r2);
fvm_mat* fvm_mat_swap_rows(fvm_mat* m, unsigned int r1, unsigned int r2);
int fvm_mat_swap_cols_r(fvm_mat* m, unsigned int c1, unsigned int c2);
fvm_mat* fvm_mat_swap_cols(fvm_mat* m, unsigned int c1, unsigned int c2);
fvm_mat* fvm_mat_transpose(fvm_mat* m);
fvm_mat* fvm_mat_transpose_mt(fvm_mat* m);

/* --- Accesing functions ---*/
fvm_mat *fvm_mat_rows_get(fvm_mat *m, unsigned int row_n);
fvm_mat *fvm_mat_col_get(fvm_mat *m, unsigned int col_n);

/* --- Matrix algebra ---- */

int fvm_mat_row_mul_r(fvm_mat *m, unsigned int row_n, double val);
fvm_mat *fvm_mat_row_mul(fvm_mat *m, unsigned int row_n, double val);
int fvm_mat_col_mul_r(fvm_mat *m, unsigned int col_n, double val);
fvm_mat* fvm_mat_col_mul(fvm_mat* m, unsigned int col_n, double val);
int fvm_mat_row_div_r(fvm_mat *m, unsigned int row_n, double a);
fvm_mat *fvm_mat_row_div(fvm_mat *m, unsigned int row_n, double a);
int fvm_mat_col_div_r(fvm_mat *m, unsigned int col_n, double val);
fvm_mat* fvm_mat_col_div(fvm_mat* m, unsigned int col_n, double val);
int fvm_mat_add_r(fvm_mat* m1, fvm_mat* m2);
fvm_mat* fvm_mat_add(fvm_mat* m1, fvm_mat* m2);
int fvm_mat_scalar_mul_r(fvm_mat* m, double val);
fvm_mat* fvm_mat_scalar_mul(fvm_mat* m, double val);
fvm_mat* fvm_mat_prod_naive(fvm_mat* m1, fvm_mat* m2);
fvm_mat* fvm_mat_prod_mt(fvm_mat* m1, fvm_mat* m2);
fvm_mat* fvm_mat_prod_cubicle_soft_6(fvm_mat* m1, fvm_mat* m2);

/* --- Miscelanious functions --- */
double fvm_rand_intrvl(double min, double max);

/* --- Creator, destroyer, checkers, basic utilities --- */

fvm_mat *fvm_mat_creator(unsigned int num_rows, unsigned int num_cols)
{
    if (num_rows == 0 || num_cols == 0) {
        CFD_ERROR(CFD_INVALID_DIMENSIONS);
        return NULL;
    }

    fvm_mat *m = calloc(1, sizeof *m);
    if (m == NULL) {
        CFD_ERROR(CFD_NULL_POINTER);
        return NULL;
    }

    m->num_rows = num_rows;
    m->num_cols = num_cols;
    m->is_square = (num_rows == num_cols);

    size_t rows = (size_t)num_rows;
    size_t cols = (size_t)num_cols;

    if (rows > 2147483647 / cols || rows * cols > 2147483647 / sizeof *m->data) {
        free(m);
        CFD_ERROR(CFD_INVALID_DIMENSIONS);
        return NULL;
    }

    m->data = calloc(rows * cols, sizeof *m->data);
    if (m->data == NULL) {
        free(m);
        return NULL;
    }

    return m;
}

fvm_mat *fvm_mat_decl(const double *arr, unsigned int num_rows, unsigned int num_cols)
{
    fvm_mat* m = fvm_mat_creator(num_rows, num_cols);
    NULL_CHECK_PTR(m);

    size_t size = (size_t)num_rows * (size_t)num_cols;

    for (size_t i = 0; i < size; i++)
    {
        m->data[i] = arr[i];
    }
    return m;
}

void fvm_mat_destroyer(fvm_mat *m)
{
    NULL_CHECK_VOID(m);
    free(m->data);
    free(m);
}

int fvm_mat_nonzero_count(fvm_mat *m, double tol)
{
    NULL_CHECK_INT(m);

    if (tol < 0.0) {
        CFD_ERROR(CFD_INVALID_DIMENSIONS);
        return 0;
    }

    size_t total_elements = (size_t)m->num_rows * (size_t)m->num_cols;
    size_t num = 0;

    for (size_t i = 0; i < total_elements; i++) {
        if (fabs(m->data[i]) > tol) {
            num++;
        }
    }

    if (num > (size_t)INT_MAX) {
        CFD_ERROR(CFD_INVALID_DIMENSIONS);
        return 0;
    }

    return (int)num;
}

fvm_mat *fvm_mat_rand(unsigned int num_rows, unsigned int num_cols, double min, double max)
{
    fvm_mat *r = fvm_mat_creator(num_rows, num_cols);
    NULL_CHECK_PTR(r);

    srand((unsigned int)time(NULL));
    for (size_t i = 0; i < (size_t)num_cols * num_rows; i++) {
        r->data[i] = fvm_rand_intrvl(min, max);
    }

    return r;

}
fvm_mat *fvm_mat_square(unsigned int num)
{
    if (num == 0) {
        CFD_ERROR(CFD_INVALID_DIMENSIONS);
        return NULL;
    }

    return fvm_mat_creator(num, num);
}

fvm_mat *fvm_mat_square_rand(unsigned int num, double min, double max)
{
    if (min > max) {
        CFD_ERROR(CFD_OUT_OF_RANGE);
        return NULL;
    }

    fvm_mat *sqrand = fvm_mat_square(num);
    NULL_CHECK_PTR(sqrand);

    size_t total = (size_t)num * (size_t)num;
    for (size_t i = 0; i < total; i++) {
        sqrand->data[i] = fvm_rand_intrvl(min, max);
    }

    return sqrand;
}

fvm_mat *fvm_mat_eye(unsigned int num)
{
    fvm_mat *iden = fvm_mat_square(num);
    NULL_CHECK_PTR(iden);

    for (size_t i = 0; i < (size_t)num; i++) {
        iden->data[i * ((size_t)num + 1)] = 1.0;
    }

    return iden;
}

fvm_mat *fvm_mat_ones(unsigned int num_rows, unsigned int num_cols)
{
    fvm_mat *ones = fvm_mat_creator(num_rows, num_cols);
    NULL_CHECK_PTR(ones);
    size_t total = (size_t)num_rows * (size_t)num_cols;
    for (size_t i = 0; i < total; i++) {
        ones->data[i] = 1.0;
    }
    return ones;
}

fvm_mat *fvm_mat_full(unsigned int num_rows, unsigned int num_cols, double a)
{
    fvm_mat *full = fvm_mat_creator(num_rows, num_cols);
    NULL_CHECK_PTR(full);
    size_t total = (size_t)num_rows * (size_t)num_cols;
    for (size_t i = 0; i < total; i++) {
        full->data[i] = a;
    }
    return full;
}

fvm_mat *fvm_mat_diag(unsigned int num_rows, unsigned int num_cols, double val)
{
    fvm_mat *diag = fvm_mat_creator(num_rows, num_cols);
    NULL_CHECK_PTR(diag);

    size_t n = (num_rows < num_cols) ? num_rows : num_cols;
    for (size_t i = 0; i < n; i++) {
        diag->data[i * (size_t)num_cols + i] = val;
    }

    return diag;
}

fvm_mat *fvm_mat_copy(fvm_mat *m)
{
    NULL_CHECK_PTR(m);

    fvm_mat *mcopy = fvm_mat_creator(m->num_rows, m->num_cols);
    NULL_CHECK_PTR(mcopy);

    size_t total = (size_t)m->num_rows * (size_t)m->num_cols;
    memcpy(mcopy->data, m->data, total * sizeof *m->data);

    return mcopy;
}

/* --- Shape, size and matrix alternatiion and manipulation functions --- */

fvm_mat* fvm_mat_remove_row(fvm_mat* m, unsigned int row_n)
{
    NULL_CHECK_PTR(m);
    MAT_ROW_RANGE_CHECK_PTR(m, row_n);

    fvm_mat* out = fvm_mat_creator(m->num_rows - 1, m->num_cols);
    NULL_CHECK_PTR(out);

    size_t dst_row = 0;
    for (size_t src_row = 0; src_row < m->num_rows; src_row++) {
        if (src_row == row_n) {
            continue;
        }
        for (size_t col = 0; col < m->num_cols; col++) {
            MAT_AT(out, dst_row, col) = MAT_AT(m, src_row, col);
        }
        dst_row++;
    }

    return out;
}

fvm_mat* fvm_mat_remove_col(fvm_mat* m, unsigned int col_n)
{
    NULL_CHECK_PTR(m);
    MAT_COL_RANGE_CHECK_PTR(m, col_n);

    fvm_mat* out = fvm_mat_creator(m->num_rows, m->num_cols - 1);
    NULL_CHECK_PTR(out);

    for (size_t i = 0; i < m->num_rows; i++) {
        size_t dst = 0;
        for (size_t j = 0; j < m->num_cols; j++) {
            if (j == col_n) continue;
            MAT_AT(out, i, dst) = MAT_AT(m, i, j);
            dst++;
        }
    }

    return out;
}

int fvm_mat_swap_rows_r(fvm_mat* m, unsigned int r1, unsigned int r2)
{
    NULL_CHECK_INT(m);
    MAT_2ROW_RANGE_CHECK_INT(m, r1, r2);

    if (r1 == r2) return 1;

    for (size_t j = 0; j < m->num_cols; j++) {
        double tmp = MAT_AT(m, r1, j);
        MAT_AT(m, r1, j) = MAT_AT(m, r2, j);
        MAT_AT(m, r2, j) = tmp;
    }

    return 1;
}

fvm_mat* fvm_mat_swap_rows(fvm_mat* m, unsigned int r1, unsigned int r2)
{
    NULL_CHECK_PTR(m);
    MAT_2ROW_RANGE_CHECK_PTR(m, r1, r2);

    fvm_mat* out = fvm_mat_copy(m);
    NULL_CHECK_PTR(out);

    for (size_t j = 0; j < out->num_cols; j++) {
        double tmp = MAT_AT(out, r1, j);
        MAT_AT(out, r1, j) = MAT_AT(out, r2, j);
        MAT_AT(out, r2, j) = tmp;
    }

    return out;
}

int fvm_mat_swap_cols_r(fvm_mat* m, unsigned int c1, unsigned int c2)
{
    NULL_CHECK_INT(m);
    MAT_2COL_RANGE_CHECK_INT(m, c1, c2);

    if (c1 == c2) return 1;

    for (size_t i = 0; i < m->num_rows; i++) {
        double tmp = MAT_AT(m, i, c1);
        MAT_AT(m, i, c1) = MAT_AT(m, i, c2);
        MAT_AT(m, i, c2) = tmp;
    }

    return 1;
}

fvm_mat* fvm_mat_swap_cols(fvm_mat* m, unsigned int c1, unsigned int c2)
{
    NULL_CHECK_PTR(m);
    MAT_2COL_RANGE_CHECK_PTR(m, c1, c2);

    fvm_mat* out = fvm_mat_copy(m);
    NULL_CHECK_PTR(out);

    for (size_t i = 0; i < out->num_rows; i++) {
        double tmp = MAT_AT(out, i, c1);
        MAT_AT(out, i, c1) = MAT_AT(out, i, c2);
        MAT_AT(out, i, c2) = tmp;
    }

    return out;
}

fvm_mat* fvm_mat_transpose(fvm_mat* m)
{
    NULL_CHECK_PTR(m);

    fvm_mat* out = fvm_mat_creator(m->num_cols, m->num_rows);
    NULL_CHECK_PTR(out);

    for (size_t i = 0; i < m->num_rows; i++) {
        for (size_t j = 0; j < m->num_cols; j++) {
            MAT_AT(out, j, i) = MAT_AT(m, i, j);
        }
    }

    return out;
}

fvm_mat* fvm_mat_transpose_mt(fvm_mat* m)
{
    NULL_CHECK_PTR(m);

    fvm_mat* out = fvm_mat_creator(m->num_cols, m->num_rows);
    NULL_CHECK_PTR(out);

    #pragma omp parallel for
    for (size_t i = 0; i < m->num_rows; i++) {
        for (size_t j = 0; j < m->num_cols; j++) {
            MAT_AT(out, j, i) = MAT_AT(m, i, j);
        }
    }

    return out;
}


/* --- Accesing functions ---*/

fvm_mat *fvm_mat_rows_get(fvm_mat *m, unsigned int row_n)
{
    NULL_CHECK_PTR(m);

    if (row_n >= m->num_rows) {
        CFD_ERROR(CFD_OUT_OF_RANGE);
        return NULL;
    }

    fvm_mat *r = fvm_mat_creator(1, m->num_cols);
    NULL_CHECK_PTR(r);
    size_t offset = (size_t)row_n * m->num_cols;
    memcpy(r->data, &m->data[offset], m->num_cols * sizeof *r->data);
    return r;
}

fvm_mat *fvm_mat_col_get(fvm_mat *m, unsigned int col_n)
{
    NULL_CHECK_PTR(m);

    if (col_n >= m->num_cols) {
        CFD_ERROR(CFD_OUT_OF_RANGE);
        return NULL;
    }

    fvm_mat *c = fvm_mat_creator(m->num_rows, 1);
    NULL_CHECK_PTR(c);

    for (size_t i = 0; i < m->num_rows; i++) {
        c->data[i] = MAT_AT(m, i, col_n);
    }

    return c;
}

/* --- Matrix algebra --- */

int fvm_mat_row_mul_r(fvm_mat *m, unsigned int row_n, double val)
{
    NULL_CHECK_INT(m);

    if (row_n >= m->num_rows) {
        CFD_ERROR(CFD_OUT_OF_RANGE);
        return 0;
    }

    for (size_t i = 0; i < m->num_cols; i++) {
        MAT_AT(m, row_n, i) *= val;
    }

    return 1;
}

fvm_mat *fvm_mat_row_mul(fvm_mat *m, unsigned int row_n, double val)
{
    NULL_CHECK_PTR(m);

    if (row_n >= m->num_rows) {
        CFD_ERROR(CFD_OUT_OF_RANGE);
        return NULL;
    }

    fvm_mat *mcopy = fvm_mat_copy(m);
    NULL_CHECK_PTR(mcopy);

    for (size_t j = 0; j < mcopy->num_cols; j++) {
        MAT_AT(mcopy, row_n, j) *= val;
    }

    return mcopy;
}

int fvm_mat_col_mul_r(fvm_mat *m, unsigned int col_n, double val)
{
    NULL_CHECK_INT(m);

    if (col_n >= m->num_cols) {
        CFD_ERROR(CFD_OUT_OF_RANGE);
        return 0;
    }

    for (size_t i = 0; i < m->num_rows; i++) {
        MAT_AT(m, i, col_n) *= val;
    }

    return 1;
}

fvm_mat* fvm_mat_col_mul(fvm_mat* m, unsigned int col_n, double val)
{
    NULL_CHECK_PTR(m);
    MAT_COL_RANGE_CHECK_PTR(m, col_n);

    fvm_mat* mcopy = fvm_mat_copy(m);
    NULL_CHECK_PTR(mcopy);

    for (size_t i = 0; i < mcopy->num_rows; i++) {
        MAT_AT(mcopy, i, col_n) *= val;
    }

    return mcopy;
}

int fvm_mat_row_div_r(fvm_mat* m, unsigned int row_n, double a)
{
    NULL_CHECK_INT(m);

    if (row_n < 0 || row_n >= m->num_rows) {
        CFD_ERROR(CFD_OUT_OF_RANGE);
        return 0;
    }

    DIV_0_CHECK_INT(a);

    int n_cols = m->num_cols;

    for (size_t j = 0; j < n_cols; j++)
    {
        MAT_AT(m, row_n, j) /= a;
    }

    return 1;
}

fvm_mat* fvm_mat_row_div(fvm_mat* m, unsigned int row_n, double a)
{
    NULL_CHECK_PTR(m);

    if (row_n < 0 || row_n >= m->num_rows) {
        CFD_ERROR(CFD_OUT_OF_RANGE);
        return NULL;
    }

    DIV_0_CHECK_PTR(a);

    fvm_mat* mcopy = fvm_mat_copy(m);
    NULL_CHECK_PTR(mcopy);

    int n_cols = mcopy->num_cols;

    for (size_t j = 0; j < n_cols; j++)
    {
        MAT_AT(mcopy, row_n, j) /= a;
    }

    return mcopy;
}

int fvm_mat_col_div_r(fvm_mat *m, unsigned int col_n, double val)
{
    NULL_CHECK_INT(m);
    MAT_COL_RANGE_CHECK_INT(m, col_n);
    DIV_0_CHECK_INT(val);

    for (size_t i = 0; i < m->num_rows; i++) {
        MAT_AT(m, i, col_n) /= val;
    }

    return 1;
}

fvm_mat* fvm_mat_col_div(fvm_mat* m, unsigned int col_n, double val)
{
    NULL_CHECK_PTR(m);
    MAT_COL_RANGE_CHECK_PTR(m, col_n);
    DIV_0_CHECK_PTR(val);

    fvm_mat* mcopy = fvm_mat_copy(m);
    NULL_CHECK_PTR(mcopy);

    for (size_t i = 0; i < mcopy->num_rows; i++) {
        MAT_AT(mcopy, i, col_n) /= val;
    }

    return mcopy;
}

int fvm_mat_add_r(fvm_mat* m1, fvm_mat* m2)
{
    NULL_CHECK_INT(m1);
    NULL_CHECK_INT(m2);
    MAT_DIM_COMPARE_ADD_INT(m1, m2);

    size_t n = (size_t)m1->num_rows * m1->num_cols;
    for (size_t i = 0; i < n; i++) {
        m1->data[i] += m2->data[i];
    }

    return 1;
}

fvm_mat* fvm_mat_add(fvm_mat* m1, fvm_mat* m2)
{
    NULL_CHECK_PTR(m1);
    NULL_CHECK_PTR(m2);
    MAT_DIM_COMPARE_ADD_NULL(m1, m2);

    fvm_mat* mcopy = fvm_mat_copy(m1);
    NULL_CHECK_PTR(mcopy);

    size_t n = (size_t)mcopy->num_rows * mcopy->num_cols;
    for (size_t i = 0; i < n; i++) {
        mcopy->data[i] += m2->data[i];
    }

    return mcopy;
}

int fvm_mat_scalar_mul_r(fvm_mat* m, double val)
{
    NULL_CHECK_INT(m);

    size_t n = (size_t)m->num_rows * m->num_cols;
    for (size_t i = 0; i < n; i++) {
        m->data[i] *= val;
    }

    return 1;
}

fvm_mat* fvm_mat_scalar_mul(fvm_mat* m, double val)
{
    NULL_CHECK_PTR(m);

    fvm_mat* mcopy = fvm_mat_copy(m);
    NULL_CHECK_PTR(mcopy);

    size_t n = (size_t)mcopy->num_rows * mcopy->num_cols;
    for (size_t i = 0; i < n; i++) {
        mcopy->data[i] *= val;
    }

    return mcopy;
}

fvm_mat* fvm_mat_prod_naive(fvm_mat* m1, fvm_mat* m2)
{
    NULL_CHECK_PTR(m1);
    NULL_CHECK_PTR(m2);
    MAT_DIM_COMPARE_MULT_NULL(m1, m2);

    fvm_mat* res = fvm_mat_creator(m1->num_rows, m2->num_cols);
    NULL_CHECK_PTR(res);

    for (size_t i = 0; i < m1->num_rows; i++) {
        for (size_t j = 0; j < m2->num_cols; j++) {
            double sum = 0.0;
            for (size_t k = 0; k < m1->num_cols; k++) {
                sum += MAT_AT(m1, i, k) * MAT_AT(m2, k, j);
            }
            MAT_AT(res, i, j) = sum;
        }
    }

    return res;
}

fvm_mat* fvm_mat_prod_mt(fvm_mat* m1, fvm_mat* m2)
{
    NULL_CHECK_PTR(m1);
    NULL_CHECK_PTR(m2);
    MAT_DIM_COMPARE_MULT_NULL(m1, m2);

    fvm_mat* res = fvm_mat_creator(m1->num_rows, m2->num_cols);
    NULL_CHECK_PTR(res);

    #pragma omp parallel for
    for (size_t i = 0; i < m1->num_rows; i++) {
        for (size_t k = 0; k < m1->num_cols; k++) {
            double a = MAT_AT(m1, i, k);
            for (size_t j = 0; j < m2->num_cols; j++) {
                MAT_AT(res, i, j) += a * MAT_AT(m2, k, j);
            }
        }
    }

    return res;
}

fvm_mat *fvm_mat_prod_cubicle_soft_6(fvm_mat *m1, fvm_mat *m2)
{
    NULL_CHECK_PTR(m1);
    NULL_CHECK_PTR(m2);
    MAT_DIM_COMPARE_MULT_NULL(m1, m2);

    size_t rows = (size_t)m1->num_rows;
    size_t inner = (size_t)m1->num_cols;
    size_t cols = (size_t)m2->num_cols;

    fvm_mat *res = fvm_mat_creator(m1->num_rows, m2->num_cols);
    NULL_CHECK_PTR(res);

    for (size_t i = 0; i < rows; i++) {
        double tmpa = m1->data[i * inner];
        for (size_t j = 0; j < cols; j++) {
            res->data[i * cols + j] = tmpa * m2->data[j];
        }

        for (size_t k = 1; k < inner; k++) {
            tmpa = m1->data[i * inner + k];
            for (size_t j = 0; j < cols; j++) {
                res->data[i * cols + j] += tmpa * m2->data[k * cols + j];
            }
        }
    }

    return res;
}

/*Miscelanious utilities*/
double fvm_rand_intrvl(double min, double max)
{
    double d = (double) rand() / (double) RAND_MAX;
    return min + d * (max - min);
}