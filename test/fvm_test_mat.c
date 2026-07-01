#include "../include/CFDlinalg/fvm.h"
#include "../include/CFDlinalg/fvm_mat.h"
#include "../include/CFDlinalg/fvm_vec.h"
#include <stdlib.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <CUnit/Basic.h>
#include <CUnit/CUnit.h>

#define TEST_ERROR_CLEANUP do { \
    CU_cleanup_registry(); \
    return CU_get_error(); \
} while (0)

static void assert_matrix_equal(const fvm_mat *a, const fvm_mat *b, double tol)
{
    CU_ASSERT_PTR_NOT_NULL(a);
    CU_ASSERT_PTR_NOT_NULL(b);
    if (a == NULL || b == NULL) return;

    CU_ASSERT_EQUAL(a->num_rows, b->num_rows);
    CU_ASSERT_EQUAL(a->num_cols, b->num_cols);

    size_t n = (size_t)a->num_rows * (size_t)a->num_cols;
    for (size_t i = 0; i < n; ++i) {
        CU_ASSERT_DOUBLE_EQUAL(a->data[i], b->data[i], tol);
    }
}

void test_mat_creator_destroyer(void)
{
    unsigned int nr = 2, nc = 3;
    fvm_mat *m = fvm_mat_creator(nr, nc);

    CU_ASSERT_PTR_NOT_NULL(m);
    if (m == NULL) return;

    CU_ASSERT_EQUAL(m->num_rows, nr);
    CU_ASSERT_EQUAL(m->num_cols, nc);
    CU_ASSERT_EQUAL(m->is_square, 0);

    size_t n = (size_t)nr * (size_t)nc;
    for (size_t i = 0; i < n; ++i) {
        CU_ASSERT_DOUBLE_EQUAL(m->data[i], 0.0, 1e-12);
    }

    fvm_mat_destroyer(m);
}

void test_mat_decl(void)
{
    double data[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    fvm_mat *m = fvm_mat_decl(data, 2, 3);
    fvm_mat *exp = fvm_mat_decl(data, 2, 3);

    assert_matrix_equal(m, exp, 1e-12);

    fvm_mat_destroyer(m);
    fvm_mat_destroyer(exp);
}

void test_mat_nonzero_count(void)
{
    double data[] = {0.0, 1e-13, -2.0, 0.0, 3.0, 1e-9};
    fvm_mat *m = fvm_mat_decl(data, 2, 3);
    int r = fvm_mat_nonzero_count(m, 1e-12);
    CU_ASSERT_EQUAL(r, 3);
    fvm_mat_destroyer(m);
}

void test_mat_square(void)
{
    fvm_mat *m = fvm_mat_square(3);
    CU_ASSERT_PTR_NOT_NULL(m);
    if (m == NULL) return;
    CU_ASSERT_EQUAL(m->num_rows, 3);
    CU_ASSERT_EQUAL(m->num_cols, 3);
    CU_ASSERT_EQUAL(m->is_square, 1);
    fvm_mat_destroyer(m);
}

void test_mat_eye(void)
{
    fvm_mat *m = fvm_mat_eye(3);
    double exp_data[] = {1.0, 0.0, 0.0,
                         0.0, 1.0, 0.0,
                         0.0, 0.0, 1.0};
    fvm_mat *exp = fvm_mat_decl(exp_data, 3, 3);

    assert_matrix_equal(m, exp, 1e-12);

    fvm_mat_destroyer(m);
    fvm_mat_destroyer(exp);
}

void test_mat_ones(void)
{
    fvm_mat *m = fvm_mat_ones(2, 3);
    double exp_data[] = {1.0, 1.0, 1.0,
                         1.0, 1.0, 1.0};
    fvm_mat *exp = fvm_mat_decl(exp_data, 2, 3);
    assert_matrix_equal(m, exp, 1e-12);
    fvm_mat_destroyer(m);
    fvm_mat_destroyer(exp);
}

void test_mat_full(void)
{
    fvm_mat *m = fvm_mat_full(2, 2, -2.5);
    double exp_data[] = {-2.5, -2.5,
                         -2.5, -2.5};
    fvm_mat *exp = fvm_mat_decl(exp_data, 2, 2);
    assert_matrix_equal(m, exp, 1e-12);
    fvm_mat_destroyer(m);
    fvm_mat_destroyer(exp);
}

void test_mat_diag(void)
{
    fvm_mat *m = fvm_mat_diag(2, 3, 7.0);
    double exp_data[] = {7.0, 0.0, 0.0,
                         0.0, 7.0, 0.0};
    fvm_mat *exp = fvm_mat_decl(exp_data, 2, 3);
    assert_matrix_equal(m, exp, 1e-12);
    fvm_mat_destroyer(m);
    fvm_mat_destroyer(exp);
}

void test_mat_copy(void)
{
    double data[] = {1.0, 2.0, 3.0, 4.0};
    fvm_mat *m = fvm_mat_decl(data, 2, 2);
    fvm_mat *cpy = fvm_mat_copy(m);

    assert_matrix_equal(m, cpy, 1e-12);
    CU_ASSERT_PTR_NOT_EQUAL(m, cpy);

    fvm_mat_destroyer(m);
    fvm_mat_destroyer(cpy);
}

void test_mat_remove_row(void)
{
    double data[] = {1.0, 2.0,
                     3.0, 4.0,
                     5.0, 6.0};
    fvm_mat *m = fvm_mat_decl(data, 3, 2);
    fvm_mat *r = fvm_mat_remove_row(m, 1);
    double exp_data[] = {1.0, 2.0,
                         5.0, 6.0};
    fvm_mat *exp = fvm_mat_decl(exp_data, 2, 2);

    assert_matrix_equal(r, exp, 1e-12);

    fvm_mat_destroyer(m);
    fvm_mat_destroyer(r);
    fvm_mat_destroyer(exp);
}

void test_mat_remove_col(void)
{
    double data[] = {1.0, 2.0, 3.0,
                     4.0, 5.0, 6.0};
    fvm_mat *m = fvm_mat_decl(data, 2, 3);
    fvm_mat *r = fvm_mat_remove_col(m, 1);
    double exp_data[] = {1.0, 3.0,
                         4.0, 6.0};
    fvm_mat *exp = fvm_mat_decl(exp_data, 2, 2);

    assert_matrix_equal(r, exp, 1e-12);

    fvm_mat_destroyer(m);
    fvm_mat_destroyer(r);
    fvm_mat_destroyer(exp);
}

void test_mat_swap_rows(void)
{
    double data[] = {1.0, 2.0,
                     3.0, 4.0,
                     5.0, 6.0};
    fvm_mat *m = fvm_mat_decl(data, 3, 2);
    fvm_mat *r = fvm_mat_swap_rows(m, 0, 2);
    double exp_data[] = {5.0, 6.0,
                         3.0, 4.0,
                         1.0, 2.0};
    fvm_mat *exp = fvm_mat_decl(exp_data, 3, 2);

    assert_matrix_equal(r, exp, 1e-12);
    assert_matrix_equal(m, fvm_mat_decl(data, 3, 2), 1e-12);

    fvm_mat_destroyer(m);
    fvm_mat_destroyer(r);
    fvm_mat_destroyer(exp);
}

void test_mat_swap_rows_r(void)
{
    double data[] = {1.0, 2.0,
                     3.0, 4.0,
                     5.0, 6.0};
    double exp_data[] = {5.0, 6.0,
                         3.0, 4.0,
                         1.0, 2.0};
    fvm_mat *m = fvm_mat_decl(data, 3, 2);
    fvm_mat *exp = fvm_mat_decl(exp_data, 3, 2);

    int ret = fvm_mat_swap_rows_r(m, 0, 2);
    CU_ASSERT_EQUAL(ret, 1);
    assert_matrix_equal(m, exp, 1e-12);

    fvm_mat_destroyer(m);
    fvm_mat_destroyer(exp);
}

void test_mat_swap_cols(void)
{
    double data[] = {1.0, 2.0, 3.0,
                     4.0, 5.0, 6.0};
    fvm_mat *m = fvm_mat_decl(data, 2, 3);
    fvm_mat *r = fvm_mat_swap_cols(m, 0, 2);
    double exp_data[] = {3.0, 2.0, 1.0,
                         6.0, 5.0, 4.0};
    fvm_mat *exp = fvm_mat_decl(exp_data, 2, 3);

    assert_matrix_equal(r, exp, 1e-12);

    fvm_mat_destroyer(m);
    fvm_mat_destroyer(r);
    fvm_mat_destroyer(exp);
}

void test_mat_swap_cols_r(void)
{
    double data[] = {1.0, 2.0, 3.0,
                     4.0, 5.0, 6.0};
    double exp_data[] = {3.0, 2.0, 1.0,
                         6.0, 5.0, 4.0};
    fvm_mat *m = fvm_mat_decl(data, 2, 3);
    fvm_mat *exp = fvm_mat_decl(exp_data, 2, 3);

    int ret = fvm_mat_swap_cols_r(m, 0, 2);
    CU_ASSERT_EQUAL(ret, 1);
    assert_matrix_equal(m, exp, 1e-12);

    fvm_mat_destroyer(m);
    fvm_mat_destroyer(exp);
}

void test_mat_transpose(void)
{
    double data[] = {1.0, 2.0, 3.0,
                     4.0, 5.0, 6.0};
    fvm_mat *m = fvm_mat_decl(data, 2, 3);
    fvm_mat *r = fvm_mat_transpose(m);
    double exp_data[] = {1.0, 4.0,
                         2.0, 5.0,
                         3.0, 6.0};
    fvm_mat *exp = fvm_mat_decl(exp_data, 3, 2);

    assert_matrix_equal(r, exp, 1e-12);

    fvm_mat_destroyer(m);
    fvm_mat_destroyer(r);
    fvm_mat_destroyer(exp);
}

void test_mat_rows_get(void)
{
    double data[] = {1.0, 2.0, 3.0,
                     4.0, 5.0, 6.0};
    fvm_mat *m = fvm_mat_decl(data, 2, 3);
    fvm_mat *r = fvm_mat_rows_get(m, 1);
    double exp_data[] = {4.0, 5.0, 6.0};
    fvm_mat *exp = fvm_mat_decl(exp_data, 1, 3);

    assert_matrix_equal(r, exp, 1e-12);

    fvm_mat_destroyer(m);
    fvm_mat_destroyer(r);
    fvm_mat_destroyer(exp);
}

void test_mat_col_get(void)
{
    double data[] = {1.0, 2.0, 3.0,
                     4.0, 5.0, 6.0};
    fvm_mat *m = fvm_mat_decl(data, 2, 3);
    fvm_mat *r = fvm_mat_col_get(m, 1);
    double exp_data[] = {2.0,
                         5.0};
    fvm_mat *exp = fvm_mat_decl(exp_data, 2, 1);

    assert_matrix_equal(r, exp, 1e-12);

    fvm_mat_destroyer(m);
    fvm_mat_destroyer(r);
    fvm_mat_destroyer(exp);
}

void test_mat_row_mul(void)
{
    double data[] = {1.0, 2.0,
                     3.0, 4.0};
    fvm_mat *m = fvm_mat_decl(data, 2, 2);
    fvm_mat *r = fvm_mat_row_mul(m, 1, -2.0);
    double exp_data[] = {1.0, 2.0,
                         -6.0, -8.0};
    fvm_mat *exp = fvm_mat_decl(exp_data, 2, 2);

    assert_matrix_equal(r, exp, 1e-12);

    fvm_mat_destroyer(m);
    fvm_mat_destroyer(r);
    fvm_mat_destroyer(exp);
}

void test_mat_row_mul_r(void)
{
    double data[] = {1.0, 2.0,
                     3.0, 4.0};
    double exp_data[] = {1.0, 2.0,
                         -6.0, -8.0};
    fvm_mat *m = fvm_mat_decl(data, 2, 2);
    fvm_mat *exp = fvm_mat_decl(exp_data, 2, 2);

    int ret = fvm_mat_row_mul_r(m, 1, -2.0);
    CU_ASSERT_EQUAL(ret, 1);
    assert_matrix_equal(m, exp, 1e-12);

    fvm_mat_destroyer(m);
    fvm_mat_destroyer(exp);
}

void test_mat_col_mul(void)
{
    double data[] = {1.0, 2.0,
                     3.0, 4.0};
    fvm_mat *m = fvm_mat_decl(data, 2, 2);
    fvm_mat *r = fvm_mat_col_mul(m, 0, 3.0);
    double exp_data[] = {3.0, 2.0,
                         9.0, 4.0};
    fvm_mat *exp = fvm_mat_decl(exp_data, 2, 2);

    assert_matrix_equal(r, exp, 1e-12);

    fvm_mat_destroyer(m);
    fvm_mat_destroyer(r);
    fvm_mat_destroyer(exp);
}

void test_mat_col_mul_r(void)
{
    double data[] = {1.0, 2.0,
                     3.0, 4.0};
    double exp_data[] = {3.0, 2.0,
                         9.0, 4.0};
    fvm_mat *m = fvm_mat_decl(data, 2, 2);
    fvm_mat *exp = fvm_mat_decl(exp_data, 2, 2);

    int ret = fvm_mat_col_mul_r(m, 0, 3.0);
    CU_ASSERT_EQUAL(ret, 1);
    assert_matrix_equal(m, exp, 1e-12);

    fvm_mat_destroyer(m);
    fvm_mat_destroyer(exp);
}

void test_mat_row_div(void)
{
    double data[] = {2.0, 4.0,
                     6.0, 8.0};
    fvm_mat *m = fvm_mat_decl(data, 2, 2);
    fvm_mat *r = fvm_mat_row_div(m, 0, 2.0);
    double exp_data[] = {1.0, 2.0,
                         6.0, 8.0};
    fvm_mat *exp = fvm_mat_decl(exp_data, 2, 2);

    assert_matrix_equal(r, exp, 1e-12);

    fvm_mat_destroyer(m);
    fvm_mat_destroyer(r);
    fvm_mat_destroyer(exp);
}

void test_mat_row_div_r(void)
{
    double data[] = {2.0, 4.0,
                     6.0, 8.0};
    double exp_data[] = {1.0, 2.0,
                         6.0, 8.0};
    fvm_mat *m = fvm_mat_decl(data, 2, 2);
    fvm_mat *exp = fvm_mat_decl(exp_data, 2, 2);

    int ret = fvm_mat_row_div_r(m, 0, 2.0);
    CU_ASSERT_EQUAL(ret, 1);
    assert_matrix_equal(m, exp, 1e-12);

    fvm_mat_destroyer(m);
    fvm_mat_destroyer(exp);
}

void test_mat_col_div(void)
{
    double data[] = {2.0, 4.0,
                     6.0, 8.0};
    fvm_mat *m = fvm_mat_decl(data, 2, 2);
    fvm_mat *r = fvm_mat_col_div(m, 1, 4.0);
    double exp_data[] = {2.0, 1.0,
                         6.0, 2.0};
    fvm_mat *exp = fvm_mat_decl(exp_data, 2, 2);

    assert_matrix_equal(r, exp, 1e-12);

    fvm_mat_destroyer(m);
    fvm_mat_destroyer(r);
    fvm_mat_destroyer(exp);
}

void test_mat_col_div_r(void)
{
    double data[] = {2.0, 4.0,
                     6.0, 8.0};
    double exp_data[] = {2.0, 1.0,
                         6.0, 2.0};
    fvm_mat *m = fvm_mat_decl(data, 2, 2);
    fvm_mat *exp = fvm_mat_decl(exp_data, 2, 2);

    int ret = fvm_mat_col_div_r(m, 1, 4.0);
    CU_ASSERT_EQUAL(ret, 1);
    assert_matrix_equal(m, exp, 1e-12);

    fvm_mat_destroyer(m);
    fvm_mat_destroyer(exp);
}

void test_mat_add(void)
{
    double a_data[] = {1.0, 2.0,
                       3.0, 4.0};
    double b_data[] = {-1.0, 0.0,
                       5.0, 1.0};
    double exp_data[] = {0.0, 2.0,
                         8.0, 5.0};

    fvm_mat *a = fvm_mat_decl(a_data, 2, 2);
    fvm_mat *b = fvm_mat_decl(b_data, 2, 2);
    fvm_mat *exp = fvm_mat_decl(exp_data, 2, 2);
    fvm_mat *r = fvm_mat_add(a, b);

    assert_matrix_equal(r, exp, 1e-12);

    fvm_mat_destroyer(a);
    fvm_mat_destroyer(b);
    fvm_mat_destroyer(exp);
    fvm_mat_destroyer(r);
}

void test_mat_add_r(void)
{
    double a_data[] = {1.0, 2.0,
                       3.0, 4.0};
    double b_data[] = {-1.0, 0.0,
                       5.0, 1.0};
    double exp_data[] = {0.0, 2.0,
                         8.0, 5.0};

    fvm_mat *a = fvm_mat_decl(a_data, 2, 2);
    fvm_mat *b = fvm_mat_decl(b_data, 2, 2);
    fvm_mat *exp = fvm_mat_decl(exp_data, 2, 2);

    int ret = fvm_mat_add_r(a, b);
    CU_ASSERT_EQUAL(ret, 1);
    assert_matrix_equal(a, exp, 1e-12);

    fvm_mat_destroyer(a);
    fvm_mat_destroyer(b);
    fvm_mat_destroyer(exp);
}

void test_mat_scalar_mul(void)
{
    double data[] = {1.0, -2.0,
                     3.0, -4.0};
    double exp_data[] = {-0.5, 1.0,
                         -1.5, 2.0};
    fvm_mat *m = fvm_mat_decl(data, 2, 2);
    fvm_mat *exp = fvm_mat_decl(exp_data, 2, 2);
    fvm_mat *r = fvm_mat_scalar_mul(m, -0.5);

    assert_matrix_equal(r, exp, 1e-12);

    fvm_mat_destroyer(m);
    fvm_mat_destroyer(exp);
    fvm_mat_destroyer(r);
}

void test_mat_scalar_mul_r(void)
{
    double data[] = {1.0, -2.0,
                     3.0, -4.0};
    double exp_data[] = {-0.5, 1.0,
                         -1.5, 2.0};
    fvm_mat *m = fvm_mat_decl(data, 2, 2);
    fvm_mat *exp = fvm_mat_decl(exp_data, 2, 2);

    int ret = fvm_mat_scalar_mul_r(m, -0.5);
    CU_ASSERT_EQUAL(ret, 1);
    assert_matrix_equal(m, exp, 1e-12);

    fvm_mat_destroyer(m);
    fvm_mat_destroyer(exp);
}

void test_mat_prod_naive(void)
{
    double a_data[] = {1.0, 2.0,
                       3.0, 4.0};
    double b_data[] = {2.0, 0.0,
                       1.0, 2.0};
    double exp_data[] = {4.0, 4.0,
                         10.0, 8.0};

    fvm_mat *a = fvm_mat_decl(a_data, 2, 2);
    fvm_mat *b = fvm_mat_decl(b_data, 2, 2);
    fvm_mat *exp = fvm_mat_decl(exp_data, 2, 2);
    fvm_mat *r = fvm_mat_prod_naive(a, b);

    assert_matrix_equal(r, exp, 1e-12);

    fvm_mat_destroyer(a);
    fvm_mat_destroyer(b);
    fvm_mat_destroyer(exp);
    fvm_mat_destroyer(r);
}

void test_mat_prod_cubicle_soft_6(void)
{
    double a_data[] = {1.0, 2.0,
                       3.0, 4.0};
    double b_data[] = {2.0, 0.0,
                       1.0, 2.0};
    double exp_data[] = {4.0, 4.0,
                         10.0, 8.0};

    fvm_mat *a = fvm_mat_decl(a_data, 2, 2);
    fvm_mat *b = fvm_mat_decl(b_data, 2, 2);
    fvm_mat *exp = fvm_mat_decl(exp_data, 2, 2);
    fvm_mat *r = fvm_mat_prod_cubicle_soft_6(a, b);

    assert_matrix_equal(r, exp, 1e-12);

    fvm_mat_destroyer(a);
    fvm_mat_destroyer(b);
    fvm_mat_destroyer(exp);
    fvm_mat_destroyer(r);
}

int init_suite(void)
{
    return 0;
}

int clean_suite(void)
{
    return 0;
}

int main(void)
{
    printf("Starting matrix test procedure.\n");

    if (CU_initialize_registry() != CUE_SUCCESS)
        return CU_get_error();

    CU_pSuite pSuite = CU_add_suite("Matrix lib test", init_suite, clean_suite);
    if (pSuite == NULL)
        TEST_ERROR_CLEANUP;

    if (CU_add_test(pSuite, "Test matrix creator", test_mat_creator_destroyer) == NULL) TEST_ERROR_CLEANUP;
    if (CU_add_test(pSuite, "Test matrix declaration", test_mat_decl) == NULL) TEST_ERROR_CLEANUP;
    if (CU_add_test(pSuite, "Test matrix nonzero count", test_mat_nonzero_count) == NULL) TEST_ERROR_CLEANUP;
    if (CU_add_test(pSuite, "Test square matrix creator", test_mat_square) == NULL) TEST_ERROR_CLEANUP;
    if (CU_add_test(pSuite, "Test identity matrix creator", test_mat_eye) == NULL) TEST_ERROR_CLEANUP;
    if (CU_add_test(pSuite, "Test ones matrix creator", test_mat_ones) == NULL) TEST_ERROR_CLEANUP;
    if (CU_add_test(pSuite, "Test full matrix creator", test_mat_full) == NULL) TEST_ERROR_CLEANUP;
    if (CU_add_test(pSuite, "Test diagonal matrix creator", test_mat_diag) == NULL) TEST_ERROR_CLEANUP;
    if (CU_add_test(pSuite, "Test matrix copy", test_mat_copy) == NULL) TEST_ERROR_CLEANUP;
    if (CU_add_test(pSuite, "Test remove row", test_mat_remove_row) == NULL) TEST_ERROR_CLEANUP;
    if (CU_add_test(pSuite, "Test remove column", test_mat_remove_col) == NULL) TEST_ERROR_CLEANUP;
    if (CU_add_test(pSuite, "Test swap rows", test_mat_swap_rows) == NULL) TEST_ERROR_CLEANUP;
    if (CU_add_test(pSuite, "Test swap rows in place", test_mat_swap_rows_r) == NULL) TEST_ERROR_CLEANUP;
    if (CU_add_test(pSuite, "Test swap columns", test_mat_swap_cols) == NULL) TEST_ERROR_CLEANUP;
    if (CU_add_test(pSuite, "Test swap columns in place", test_mat_swap_cols_r) == NULL) TEST_ERROR_CLEANUP;
    if (CU_add_test(pSuite, "Test transpose", test_mat_transpose) == NULL) TEST_ERROR_CLEANUP;
    if (CU_add_test(pSuite, "Test get row", test_mat_rows_get) == NULL) TEST_ERROR_CLEANUP;
    if (CU_add_test(pSuite, "Test get column", test_mat_col_get) == NULL) TEST_ERROR_CLEANUP;
    if (CU_add_test(pSuite, "Test row multiplication", test_mat_row_mul) == NULL) TEST_ERROR_CLEANUP;
    if (CU_add_test(pSuite, "Test row multiplication in place", test_mat_row_mul_r) == NULL) TEST_ERROR_CLEANUP;
    if (CU_add_test(pSuite, "Test column multiplication", test_mat_col_mul) == NULL) TEST_ERROR_CLEANUP;
    if (CU_add_test(pSuite, "Test column multiplication in place", test_mat_col_mul_r) == NULL) TEST_ERROR_CLEANUP;
    if (CU_add_test(pSuite, "Test row division", test_mat_row_div) == NULL) TEST_ERROR_CLEANUP;
    if (CU_add_test(pSuite, "Test row division in place", test_mat_row_div_r) == NULL) TEST_ERROR_CLEANUP;
    if (CU_add_test(pSuite, "Test column division", test_mat_col_div) == NULL) TEST_ERROR_CLEANUP;
    if (CU_add_test(pSuite, "Test column division in place", test_mat_col_div_r) == NULL) TEST_ERROR_CLEANUP;
    if (CU_add_test(pSuite, "Test matrix addition", test_mat_add) == NULL) TEST_ERROR_CLEANUP;
    if (CU_add_test(pSuite, "Test matrix addition in place", test_mat_add_r) == NULL) TEST_ERROR_CLEANUP;
    if (CU_add_test(pSuite, "Test matrix scalar multiplication", test_mat_scalar_mul) == NULL) TEST_ERROR_CLEANUP;
    if (CU_add_test(pSuite, "Test matrix scalar multiplication in place", test_mat_scalar_mul_r) == NULL) TEST_ERROR_CLEANUP;
    if (CU_add_test(pSuite, "Test naive matrix product", test_mat_prod_naive) == NULL) TEST_ERROR_CLEANUP;
    if (CU_add_test(pSuite, "Test cubicle soft 6 matrix product", test_mat_prod_cubicle_soft_6) == NULL) TEST_ERROR_CLEANUP;

    CU_basic_set_mode(CU_BRM_VERBOSE);
    CU_basic_run_tests();
    CU_cleanup_registry();
    return CU_get_error();
}
