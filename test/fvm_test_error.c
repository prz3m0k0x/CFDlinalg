#include "../include/CFDlinalg/fvm.h"
#include "../include/CFDlinalg/fvm_mat.h"
#include "../include/CFDlinalg/fvm_vec.h"
#include "../include/CFDlinalg/fvm_linalg.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <CUnit/Basic.h>
#include <CUnit/CUnit.h>

#define TEST_ERROR_CLEANUP do { \
    CU_cleanup_registry(); \
    return CU_get_error(); \
} while (0)

void test_vec_add_r_dim_mismatch(void)
{
    double a_data[] = {1.0, 2.0, 3.0};
    double b_data[] = {4.0, 5.0};
    fvm_vector *a = fvm_vec_decl(a_data, 3);
    fvm_vector *b = fvm_vec_decl(b_data, 2);

    int ret = fvm_vec_add_r(a, b);
    CU_ASSERT_EQUAL(ret, 0);

    fvm_vec_destroyer(a);
    fvm_vec_destroyer(b);
}

void test_vec_scalar_div_r_zero(void)
{
    double a_data[] = {1.0, 2.0, 3.0};
    fvm_vector *a = fvm_vec_decl(a_data, 3);

    int ret = fvm_vec_scalar_div_r(a, 0.0);
    CU_ASSERT_EQUAL(ret, 0);

    fvm_vec_destroyer(a);
}

void test_vec_dot_dim_mismatch(void)
{
    double a_data[] = {1.0, 2.0, 3.0};
    double b_data[] = {4.0, 5.0};
    fvm_vector *a = fvm_vec_decl(a_data, 3);
    fvm_vector *b = fvm_vec_decl(b_data, 2);

    double ret = fvm_vec_dot(a, b);
    CU_ASSERT_DOUBLE_EQUAL(ret, 0.0, 1e-12);

    fvm_vec_destroyer(a);
    fvm_vec_destroyer(b);
}

void test_vec_null_input(void)
{
    int ret = fvm_vec_add_r(NULL, NULL);
    CU_ASSERT_EQUAL(ret, 0);
}

void test_mat_add_r_dim_mismatch(void)
{
    double a_data[] = {1.0, 2.0, 3.0, 4.0};
    double b_data[] = {1.0, 2.0, 3.0,
                       4.0, 5.0, 6.0};
    fvm_mat *A = fvm_mat_decl(a_data, 2, 2);
    fvm_mat *B = fvm_mat_decl(b_data, 2, 3);

    int ret = fvm_mat_add_r(A, B);
    CU_ASSERT_EQUAL(ret, 0);

    fvm_mat_destroyer(A);
    fvm_mat_destroyer(B);
}

void test_mat_mul_dim_mismatch(void)
{
    double a_data[] = {1.0, 2.0, 3.0, 4.0};
    double b_data[] = {1.0, 2.0,
                       3.0, 4.0,
                       5.0, 6.0,
                       7.0, 8.0,
                       9.0, 10.0,
                       11.0, 12.0};
    fvm_mat *A = fvm_mat_decl(a_data, 2, 2);
    fvm_mat *B = fvm_mat_decl(b_data, 6, 2);

    fvm_mat *ret = fvm_mat_prod_cubicle_soft_6(A, B);
    CU_ASSERT_PTR_NULL(ret);

    fvm_mat_destroyer(A);
    fvm_mat_destroyer(B);
}

void test_mat_swap_rows_out_of_bounds(void)
{
    double a_data[] = {1.0, 2.0, 3.0, 4.0};
    fvm_mat *A = fvm_mat_decl(a_data, 2, 2);

    int ret = fvm_mat_swap_rows_r(A, 0, 5);
    CU_ASSERT_EQUAL(ret, 0);

    fvm_mat_destroyer(A);
}

void test_mat_null_input(void)
{
    int ret = fvm_mat_add_r(NULL, NULL);
    CU_ASSERT_EQUAL(ret, 0);
}

void test_linalg_system_creator_missing_rhs(void)
{
    double A_data[] = {4.0, 1.0,
                       2.0, 3.0};
    double x_data[] = {0.0, 0.0};

    fvm_mat *A = fvm_mat_decl(A_data, 2, 2);
    fvm_vector *x = fvm_vec_decl(x_data, 2);

    fvm_linalg_system *sys = fvm_linalg_system_creator(A, x, NULL, 0);
    CU_ASSERT_PTR_NULL(sys);

    fvm_mat_destroyer(A);
    fvm_vec_destroyer(x);
}

void test_linalg_system_creator_homogeneity_error(void)
{
    double A_data[] = {4.0, 1.0,
                       2.0, 3.0};
    double x_data[] = {0.0, 0.0};
    double b_data[] = {1.0, 2.0};

    fvm_mat *A = fvm_mat_decl(A_data, 2, 2);
    fvm_vector *x = fvm_vec_decl(x_data, 2);
    fvm_vector *b = fvm_vec_decl(b_data, 2);

    fvm_linalg_system *sys = fvm_linalg_system_creator(A, x, b, 1);
    CU_ASSERT_PTR_NULL(sys);

    fvm_mat_destroyer(A);
    fvm_vec_destroyer(x);
    fvm_vec_destroyer(b);
}

void test_linalg_system_creator_dim_mismatch(void)
{
    double A_data[] = {4.0, 1.0,
                       2.0, 3.0};
    double x_data[] = {0.0, 0.0};
    double b_data[] = {1.0, 2.0, 3.0};

    fvm_mat *A = fvm_mat_decl(A_data, 2, 2);
    fvm_vector *x = fvm_vec_decl(x_data, 2);
    fvm_vector *b = fvm_vec_decl(b_data, 3);

    fvm_linalg_system *sys = fvm_linalg_system_creator(A, x, b, 0);
    CU_ASSERT_PTR_NULL(sys);

    fvm_mat_destroyer(A);
    fvm_vec_destroyer(x);
    fvm_vec_destroyer(b);
}

void test_linalg_solve_unknown_solver(void)
{
    double A_data[] = {4.0, 1.0,
                       2.0, 3.0};
    double x_data[] = {0.0, 0.0};
    double b_data[] = {1.0, 2.0};

    fvm_mat *A = fvm_mat_decl(A_data, 2, 2);
    fvm_vector *x = fvm_vec_decl(x_data, 2);
    fvm_vector *b = fvm_vec_decl(b_data, 2);
    fvm_linalg_system *sys = fvm_linalg_system_creator(A, x, b, 0);

    int ret = fvm_linalg_solve_b(sys, 50, 1e-8, "sor");
    CU_ASSERT_EQUAL(ret, 0);

    fvm_linalg_system_destroyer(sys, 0);
    fvm_mat_destroyer(A);
    fvm_vec_destroyer(x);
    fvm_vec_destroyer(b);
}

void test_linalg_mat_vec_prod_dim_mismatch(void)
{
    double A_data[] = {1.0, 2.0,
                       3.0, 4.0};
    double x_data[] = {1.0, 2.0, 3.0};

    fvm_mat *A = fvm_mat_decl(A_data, 2, 2);
    fvm_vector *x = fvm_vec_decl(x_data, 3);

    fvm_vector *ret = fvm_mat_vec_prod(A, x);
    CU_ASSERT_PTR_NULL(ret);

    fvm_mat_destroyer(A);
    fvm_vec_destroyer(x);
}

void test_linalg_gauss_u_non_square(void)
{
    double A_data[] = {1.0, 2.0, 3.0,
                       4.0, 5.0, 6.0};
    fvm_mat *A = fvm_mat_decl(A_data, 2, 3);

    int ret = fvm_linalg_gauss_u(A);
    CU_ASSERT_EQUAL(ret, 0);

    fvm_mat_destroyer(A);
}

void test_linalg_jacobi_zero_diagonal(void)
{
    double A_data[] = {0.0, 1.0,
                       2.0, 3.0};
    double x_data[] = {0.0, 0.0};
    double b_data[] = {1.0, 2.0};

    fvm_mat *A = fvm_mat_decl(A_data, 2, 2);
    fvm_vector *x = fvm_vec_decl(x_data, 2);
    fvm_vector *b = fvm_vec_decl(b_data, 2);
    fvm_linalg_system *sys = fvm_linalg_system_creator(A, x, b, 0);

    int ret = fvm_linalg_solve_jacobi(sys, 50, 1e-8);
    CU_ASSERT_EQUAL(ret, 0);

    fvm_linalg_system_destroyer(sys, 0);
    fvm_mat_destroyer(A);
    fvm_vec_destroyer(x);
    fvm_vec_destroyer(b);
}

void test_linalg_gauss_seidel_zero_diagonal(void)
{
    double A_data[] = {4.0, 1.0,
                       2.0, 0.0};
    double x_data[] = {0.0, 0.0};
    double b_data[] = {1.0, 2.0};

    fvm_mat *A = fvm_mat_decl(A_data, 2, 2);
    fvm_vector *x = fvm_vec_decl(x_data, 2);
    fvm_vector *b = fvm_vec_decl(b_data, 2);
    fvm_linalg_system *sys = fvm_linalg_system_creator(A, x, b, 0);

    int ret = fvm_linalg_solve_gauss_seidel(sys, 50, 1e-8);
    CU_ASSERT_EQUAL(ret, 0);

    fvm_linalg_system_destroyer(sys, 0);
    fvm_mat_destroyer(A);
    fvm_vec_destroyer(x);
    fvm_vec_destroyer(b);
}

int init_suite(void) { return 0; }
int clean_suite(void) { return 0; }

int main(void)
{
    if (CU_initialize_registry() != CUE_SUCCESS)
        return CU_get_error();

    CU_pSuite pSuite = CU_add_suite("Error path tests", init_suite, clean_suite);
    if (pSuite == NULL)
        TEST_ERROR_CLEANUP;

    if (CU_add_test(pSuite, "vec add_r dim mismatch", test_vec_add_r_dim_mismatch) == NULL) TEST_ERROR_CLEANUP;
    if (CU_add_test(pSuite, "vec scalar_div_r zero", test_vec_scalar_div_r_zero) == NULL) TEST_ERROR_CLEANUP;
    if (CU_add_test(pSuite, "vec dot dim mismatch", test_vec_dot_dim_mismatch) == NULL) TEST_ERROR_CLEANUP;
    if (CU_add_test(pSuite, "vec null input", test_vec_null_input) == NULL) TEST_ERROR_CLEANUP;
    if (CU_add_test(pSuite, "mat add_r dim mismatch", test_mat_add_r_dim_mismatch) == NULL) TEST_ERROR_CLEANUP;
    if (CU_add_test(pSuite, "mat mul dim mismatch", test_mat_mul_dim_mismatch) == NULL) TEST_ERROR_CLEANUP;
    if (CU_add_test(pSuite, "mat swap rows out of bounds", test_mat_swap_rows_out_of_bounds) == NULL) TEST_ERROR_CLEANUP;
    if (CU_add_test(pSuite, "mat null input", test_mat_null_input) == NULL) TEST_ERROR_CLEANUP;
    if (CU_add_test(pSuite, "linalg creator missing rhs", test_linalg_system_creator_missing_rhs) == NULL) TEST_ERROR_CLEANUP;
    if (CU_add_test(pSuite, "linalg creator homogeneity error", test_linalg_system_creator_homogeneity_error) == NULL) TEST_ERROR_CLEANUP;
    if (CU_add_test(pSuite, "linalg creator dim mismatch", test_linalg_system_creator_dim_mismatch) == NULL) TEST_ERROR_CLEANUP;
    if (CU_add_test(pSuite, "linalg unknown solver", test_linalg_solve_unknown_solver) == NULL) TEST_ERROR_CLEANUP;
    if (CU_add_test(pSuite, "linalg mat vec dim mismatch", test_linalg_mat_vec_prod_dim_mismatch) == NULL) TEST_ERROR_CLEANUP;
    if (CU_add_test(pSuite, "linalg gauss_u non square", test_linalg_gauss_u_non_square) == NULL) TEST_ERROR_CLEANUP;
    if (CU_add_test(pSuite, "linalg jacobi zero diagonal", test_linalg_jacobi_zero_diagonal) == NULL) TEST_ERROR_CLEANUP;
    if (CU_add_test(pSuite, "linalg gauss seidel zero diagonal", test_linalg_gauss_seidel_zero_diagonal) == NULL) TEST_ERROR_CLEANUP;

    CU_basic_set_mode(CU_BRM_VERBOSE);
    CU_basic_run_tests();
    CU_cleanup_registry();
    return CU_get_error();
}