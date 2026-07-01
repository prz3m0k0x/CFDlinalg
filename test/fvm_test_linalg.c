#include "../include/CFDlinalg/fvm.h"
#include "../include/CFDlinalg/fvm_mat.h"
#include "../include/CFDlinalg/fvm_vec.h"
#include "../include/CFDlinalg/fvm_linalg.h"
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

static void assert_vector_equal(const fvm_vector *a, const fvm_vector *b, double tol)
{
    CU_ASSERT_PTR_NOT_NULL(a);
    CU_ASSERT_PTR_NOT_NULL(b);
    if (a == NULL || b == NULL) return;

    CU_ASSERT_EQUAL(a->num_dim, b->num_dim);
    for (size_t i = 0; i < a->num_dim; ++i) {
        CU_ASSERT_DOUBLE_EQUAL(a->data[i], b->data[i], tol);
    }
}

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

void test_linalg_system_creator_nonhom(void)
{
    double A_data[] = {4.0, 1.0,
                       2.0, 3.0};
    double x_data[] = {0.0, 0.0};
    double b_data[] = {1.0, 2.0};

    fvm_mat *A = fvm_mat_decl(A_data, 2, 2);
    fvm_vector *x = fvm_vec_decl(x_data, 2);
    fvm_vector *b = fvm_vec_decl(b_data, 2);

    fvm_linalg_system *sys = fvm_linalg_system_creator(A, x, b, 0);

    CU_ASSERT_PTR_NOT_NULL(sys);
    if (sys != NULL) {
        CU_ASSERT_PTR_EQUAL(sys->A, A);
        CU_ASSERT_PTR_EQUAL(sys->x, x);
        CU_ASSERT_PTR_EQUAL(sys->b, b);
        CU_ASSERT_EQUAL(sys->is_homogenous, 0);
    }

    fvm_linalg_system_destroyer(sys, 0);
    fvm_mat_destroyer(A);
    fvm_vec_destroyer(x);
    fvm_vec_destroyer(b);
}

void test_linalg_system_creator_hom(void)
{
    double A_data[] = {2.0, 0.0,
                       0.0, 5.0};
    double x_data[] = {1.0, 1.0};

    fvm_mat *A = fvm_mat_decl(A_data, 2, 2);
    fvm_vector *x = fvm_vec_decl(x_data, 2);

    fvm_linalg_system *sys = fvm_linalg_system_creator(A, x, NULL, 1);

    CU_ASSERT_PTR_NOT_NULL(sys);
    if (sys != NULL) {
        CU_ASSERT_PTR_EQUAL(sys->A, A);
        CU_ASSERT_PTR_EQUAL(sys->x, x);
        CU_ASSERT_PTR_NULL(sys->b);
        CU_ASSERT_EQUAL(sys->is_homogenous, 1);
    }

    fvm_linalg_system_destroyer(sys, 0);
    fvm_mat_destroyer(A);
    fvm_vec_destroyer(x);
}

void test_linalg_system_check_nonhom(void)
{
    double A_data[] = {4.0, 1.0,
                       2.0, 3.0};
    double x_data[] = {0.0, 0.0};
    double b_data[] = {1.0, 2.0};

    fvm_mat *A = fvm_mat_decl(A_data, 2, 2);
    fvm_vector *x = fvm_vec_decl(x_data, 2);
    fvm_vector *b = fvm_vec_decl(b_data, 2);
    fvm_linalg_system *sys = fvm_linalg_system_creator(A, x, b, 0);

    CU_ASSERT_EQUAL(linalg_system_check(sys), 1);

    fvm_linalg_system_destroyer(sys, 0);
    fvm_mat_destroyer(A);
    fvm_vec_destroyer(x);
    fvm_vec_destroyer(b);
}

void test_linalg_system_check_hom(void)
{
    double A_data[] = {1.0, 0.0,
                       0.0, 1.0};
    double x_data[] = {0.0, 0.0};

    fvm_mat *A = fvm_mat_decl(A_data, 2, 2);
    fvm_vector *x = fvm_vec_decl(x_data, 2);
    fvm_linalg_system *sys = fvm_linalg_system_creator(A, x, NULL, 1);

    CU_ASSERT_EQUAL(linalg_system_check(sys), 1);

    fvm_linalg_system_destroyer(sys, 0);
    fvm_mat_destroyer(A);
    fvm_vec_destroyer(x);
}

void test_mat_vec_prod(void)
{
    double A_data[] = {1.0, 2.0,
                       3.0, 4.0};
    double x_data[] = {2.0, -1.0};
    double exp_data[] = {0.0, 2.0};

    fvm_mat *A = fvm_mat_decl(A_data, 2, 2);
    fvm_vector *x = fvm_vec_decl(x_data, 2);
    fvm_vector *exp = fvm_vec_decl(exp_data, 2);
    fvm_vector *r = fvm_mat_vec_prod(A, x);

    assert_vector_equal(r, exp, 1e-12);

    fvm_mat_destroyer(A);
    fvm_vec_destroyer(x);
    fvm_vec_destroyer(exp);
    fvm_vec_destroyer(r);
}

void test_residual_inf_norm(void)
{
    double r_data[] = {1.0, 2.0, 3.0};
    fvm_vector *r = fvm_vec_decl(r_data, 3);
    double val = residual_inf_norm(r);
    CU_ASSERT_DOUBLE_EQUAL(val, 1.5, 1e-12);
    fvm_vec_destroyer(r);
}

void test_find_pivot(void)
{
    double A_data[] = {1.0, 2.0,
                       5.0, 4.0,
                       3.0, 6.0};
    fvm_mat *A = fvm_mat_decl(A_data, 3, 2);

    pivot p = fvm_linalg_find_pivot(A, 0, 0);

    CU_ASSERT_EQUAL(p.n_row, 1);
    CU_ASSERT_EQUAL(p.n_col, 0);
    CU_ASSERT_DOUBLE_EQUAL(p.pivot_val, 5.0, 1e-12);

    fvm_mat_destroyer(A);
}

void test_gauss_u(void)
{
    double A_data[] = {2.0, 1.0,
                       4.0, 3.0};
    double exp_data[] = {4.0, 3.0,
                         0.0, -0.5};

    fvm_mat *A = fvm_mat_decl(A_data, 2, 2);
    fvm_mat *exp = fvm_mat_decl(exp_data, 2, 2);

    int ret = fvm_linalg_gauss_u(A);
    CU_ASSERT_EQUAL(ret, 1);
    assert_matrix_equal(A, exp, 1e-12);

    fvm_mat_destroyer(A);
    fvm_mat_destroyer(exp);
}

void test_solve_jacobi(void)
{

    double x0_data[] = {0.0, 0.0};
    double A_data[] = {
        4.0, 1.0,
        2.0, 3.0
    };
    double b_data[] = {1.0, 2.0};
    double exp_data[] = {0.1, 0.6};

    fvm_mat *A = fvm_mat_decl(A_data, 2, 2);
    fvm_vector *x = fvm_vec_decl(x0_data, 2);
    fvm_vector *b = fvm_vec_decl(b_data, 2);
    fvm_vector *exp = fvm_vec_decl(exp_data, 2);
    fvm_linalg_system *sys = fvm_linalg_system_creator(A, x, b, 0);

    int ret = fvm_linalg_solve_jacobi(sys, 200, 1e-10);
    CU_ASSERT_EQUAL(ret, 1);
    assert_vector_equal(sys->x, exp, 1e-6);

    fvm_vec_destroyer(exp);
    fvm_linalg_system_destroyer(sys, 0);
    fvm_mat_destroyer(A);
    fvm_vec_destroyer(x);
    fvm_vec_destroyer(b);
}

void test_solve_gauss_seidel(void)
{
    double A_data[] = {4.0, 1.0,
                    2.0, 3.0};
    double x0_data[] = {0.0, 0.0};
    double b_data[] = {1.0, 2.0};
    double exp_data[] = {0.1, 0.6};

    fvm_mat *A = fvm_mat_decl(A_data, 2, 2);
    fvm_vector *x = fvm_vec_decl(x0_data, 2);
    fvm_vector *b = fvm_vec_decl(b_data, 2);
    fvm_vector *exp = fvm_vec_decl(exp_data, 2);
    fvm_linalg_system *sys = fvm_linalg_system_creator(A, x, b, 0);

    int ret = fvm_linalg_solve_gauss_seidel(sys, 100, 1e-10);
    CU_ASSERT_EQUAL(ret, 1);
    assert_vector_equal(sys->x, exp, 1e-6);

    fvm_vec_destroyer(exp);
    fvm_linalg_system_destroyer(sys, 0);
    fvm_mat_destroyer(A);
    fvm_vec_destroyer(x);
    fvm_vec_destroyer(b);
}

void test_solve_jacobi_12x12(void)
{
    double A_data[] = {
        13, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
        -1, 12, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0,
         0, -1, 11, -1,  0,  0,  0,  0,  0,  0,  0,  0,
         0,  0, -1, 14, -1,  0,  0,  0,  0,  0,  0,  0,
         0,  0,  0, -1, 15, -1,  0,  0,  0,  0,  0,  0,
         0,  0,  0,  0, -1, 13, -1,  0,  0,  0,  0,  0,
         0,  0,  0,  0,  0, -1, 12, -1,  0,  0,  0,  0,
         0,  0,  0,  0,  0,  0, -1, 11, -1,  0,  0,  0,
         0,  0,  0,  0,  0,  0,  0, -1, 14, -1,  0,  0,
         0,  0,  0,  0,  0,  0,  0,  0, -1, 13, -1,  0,
         0,  0,  0,  0,  0,  0,  0,  0,  0, -1, 12, -1,
         0,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1, 16
    };

    double x0_data[] = {0,0,0,0,0,0,0,0,0,0,0,0};

    double b_data[] = {12, 10, 9, 12, 13, 11, 10, 9, 12, 11, 10, 15};

    double x_exp_data[] = {1,1,1,1,1,1,1,1,1,1,1,1};

    fvm_mat *A = fvm_mat_decl(A_data, 12, 12);
    fvm_vector *x = fvm_vec_decl(x0_data, 12);
    fvm_vector *b = fvm_vec_decl(b_data, 12);
    fvm_vector *exp = fvm_vec_decl(x_exp_data, 12);
    fvm_linalg_system *sys = fvm_linalg_system_creator(A, x, b, 0);

    int ret = fvm_linalg_solve_b(sys, 1000, 1e-10, "jacobi");
    CU_ASSERT_EQUAL(ret, 1);

    for (size_t i = 0; i < 12; ++i) {
        CU_ASSERT_DOUBLE_EQUAL(sys->x->data[i], exp->data[i], 1e-6);
    }

    fvm_vec_destroyer(exp);
    fvm_linalg_system_destroyer(sys, 0);
    fvm_mat_destroyer(A);
    fvm_vec_destroyer(x);
    fvm_vec_destroyer(b);
}
void test_solve_gauss_seidel_12x12(void)
{
    double A_data[] = {
        13, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
        -1, 12, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0,
         0, -1, 11, -1,  0,  0,  0,  0,  0,  0,  0,  0,
         0,  0, -1, 14, -1,  0,  0,  0,  0,  0,  0,  0,
         0,  0,  0, -1, 15, -1,  0,  0,  0,  0,  0,  0,
         0,  0,  0,  0, -1, 13, -1,  0,  0,  0,  0,  0,
         0,  0,  0,  0,  0, -1, 12, -1,  0,  0,  0,  0,
         0,  0,  0,  0,  0,  0, -1, 11, -1,  0,  0,  0,
         0,  0,  0,  0,  0,  0,  0, -1, 14, -1,  0,  0,
         0,  0,  0,  0,  0,  0,  0,  0, -1, 13, -1,  0,
         0,  0,  0,  0,  0,  0,  0,  0,  0, -1, 12, -1,
         0,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1, 16
    };

    double x0_data[] = {0,0,0,0,0,0,0,0,0,0,0,0};

    double b_data[] = {12, 10, 9, 12, 13, 11, 10, 9, 12, 11, 10, 15};

    double x_exp_data[] = {1,1,1,1,1,1,1,1,1,1,1,1};

    fvm_mat *A = fvm_mat_decl(A_data, 12, 12);
    fvm_vector *x = fvm_vec_decl(x0_data, 12);
    fvm_vector *b = fvm_vec_decl(b_data, 12);
    fvm_vector *exp = fvm_vec_decl(x_exp_data, 12);
    fvm_linalg_system *sys = fvm_linalg_system_creator(A, x, b, 0);

    int ret = fvm_linalg_solve_b(sys, 200, 1e-10, "gauss-seidel");
    CU_ASSERT_EQUAL(ret, 1);

    for (size_t i = 0; i < 12; ++i) {
        CU_ASSERT_DOUBLE_EQUAL(sys->x->data[i], exp->data[i], 1e-6);
    }

    fvm_vec_destroyer(exp);
    fvm_linalg_system_destroyer(sys, 0);
    fvm_mat_destroyer(A);
    fvm_vec_destroyer(x);
    fvm_vec_destroyer(b);
}

void test_solve_b_jacobi(void)
{
    double x0_data[] = {0.0, 0.0};
    double A_data[] = {
        4.0, 1.0,
        2.0, 3.0
    };
    double b_data[] = {1.0, 2.0};
    double exp_data[] = {0.1, 0.6};

    fvm_mat *A = fvm_mat_decl(A_data, 2, 2);
    fvm_vector *x = fvm_vec_decl(x0_data, 2);
    fvm_vector *b = fvm_vec_decl(b_data, 2);
    fvm_vector *exp = fvm_vec_decl(exp_data, 2);
    fvm_linalg_system *sys = fvm_linalg_system_creator(A, x, b, 0);

    int ret = fvm_linalg_solve_b(sys, 200, 1e-10, "jacobi");
    CU_ASSERT_EQUAL(ret, 1);
    assert_vector_equal(sys->x, exp, 1e-6);

    fvm_vec_destroyer(exp);
    fvm_linalg_system_destroyer(sys, 0);
    fvm_mat_destroyer(A);
    fvm_vec_destroyer(x);
    fvm_vec_destroyer(b);
}

void test_solve_b_gauss_seidel(void)
{
    double A_data[] = {4.0, 1.0,
                    2.0, 3.0};
    double x0_data[] = {0.0, 0.0};
    double b_data[] = {1.0, 2.0};
    double exp_data[] = {0.1, 0.6};

    fvm_mat *A = fvm_mat_decl(A_data, 2, 2);
    fvm_vector *x = fvm_vec_decl(x0_data, 2);
    fvm_vector *b = fvm_vec_decl(b_data, 2);
    fvm_vector *exp = fvm_vec_decl(exp_data, 2);
    fvm_linalg_system *sys = fvm_linalg_system_creator(A, x, b, 0);

    int ret = fvm_linalg_solve_b(sys, 100, 1e-10, "gauss-seidel");
    CU_ASSERT_EQUAL(ret, 1);
    assert_vector_equal(sys->x, exp, 1e-6);

    fvm_vec_destroyer(exp);
    fvm_linalg_system_destroyer(sys, 0);
    fvm_mat_destroyer(A);
    fvm_vec_destroyer(x);
    fvm_vec_destroyer(b);
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
    printf("Starting linalg test procedure.\n");

    if (CU_initialize_registry() != CUE_SUCCESS)
        return CU_get_error();

    CU_pSuite pSuite = CU_add_suite("Linalg lib test", init_suite, clean_suite);
    if (pSuite == NULL)
        TEST_ERROR_CLEANUP;

    if (CU_add_test(pSuite, "Test linalg system creator nonhomogenous", test_linalg_system_creator_nonhom) == NULL) TEST_ERROR_CLEANUP;
    if (CU_add_test(pSuite, "Test linalg system creator homogenous", test_linalg_system_creator_hom) == NULL) TEST_ERROR_CLEANUP;
    if (CU_add_test(pSuite, "Test linalg system check nonhomogenous", test_linalg_system_check_nonhom) == NULL) TEST_ERROR_CLEANUP;
    if (CU_add_test(pSuite, "Test linalg system check homogenous", test_linalg_system_check_hom) == NULL) TEST_ERROR_CLEANUP;
    if (CU_add_test(pSuite, "Test matrix vector product", test_mat_vec_prod) == NULL) TEST_ERROR_CLEANUP;
    if (CU_add_test(pSuite, "Test residual", test_residual_inf_norm) == NULL) TEST_ERROR_CLEANUP;
    if (CU_add_test(pSuite, "Test pivot search", test_find_pivot) == NULL) TEST_ERROR_CLEANUP;
    if (CU_add_test(pSuite, "Test Gaussian elimination to upper matrix", test_gauss_u) == NULL) TEST_ERROR_CLEANUP;
    if (CU_add_test(pSuite, "Test Jacobi solver", test_solve_jacobi) == NULL) TEST_ERROR_CLEANUP;
    if (CU_add_test(pSuite, "Test Gauss Seidel solver", test_solve_gauss_seidel) == NULL) TEST_ERROR_CLEANUP;
    if (CU_add_test(pSuite, "Test solver wrapper Jacobi", test_solve_b_jacobi) == NULL) TEST_ERROR_CLEANUP;
    if (CU_add_test(pSuite, "Test solver wrapper Gauss Seidel", test_solve_b_gauss_seidel) == NULL) TEST_ERROR_CLEANUP;
    if (CU_add_test(pSuite, "Test solver wrapper G-S 12x12", test_solve_gauss_seidel_12x12) == NULL) TEST_ERROR_CLEANUP;
    if (CU_add_test(pSuite, "Test solver wrapper Jacobi 12x12", test_solve_jacobi_12x12) == NULL) TEST_ERROR_CLEANUP;

    CU_basic_set_mode(CU_BRM_VERBOSE);
    CU_basic_run_tests();
    CU_cleanup_registry();
    return CU_get_error();
}
