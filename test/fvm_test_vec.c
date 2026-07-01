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

void test_vec_creator_destroyer(void)
{
    unsigned int num_dim = 3;
    fvm_vector* r = fvm_vec_creator(num_dim);
    CU_ASSERT_PTR_NOT_NULL(r);
    CU_ASSERT_EQUAL(r->num_dim, num_dim);
    printf("Succesfully created vector with %d dimensions \n", num_dim);
    fvm_vec_destroyer(r);
    return;
}

void test_vec_copy(void)
{
        /* v1: simple positive vector, 3x1 */
    double v1_data[] = {1.0, 2.0, 3.0};
    fvm_vector *v1 = fvm_vec_decl(v1_data, 3);

    fvm_vector* r = fvm_vec_copy(v1);
    CU_ASSERT_PTR_NOT_NULL(r);
    CU_ASSERT_EQUAL(r->num_dim, v1->num_dim);

    for (size_t i = 0; i < r->num_dim; i++)
    {
        CU_ASSERT_DOUBLE_EQUAL(r->data[i], v1->data[i], 1e-12);
    }
    fvm_vec_destroyer(r);
    fvm_vec_destroyer(v1);
    return;
}
/* --- copy_contents --- */
void test_vec_copy_contents(void)
{
            /* v1: simple positive vector, 3x1 */
    double v_data[] = {1.0, 2.0, 3.0};
    fvm_vector *v = fvm_vec_decl(v_data, 3);

    fvm_vector *dst = fvm_vec_creator(v->num_dim);
    CU_ASSERT_PTR_NOT_NULL(dst);
    fvm_vec_copy_contents(dst, v);
    CU_ASSERT_EQUAL(dst->num_dim, v->num_dim);
    for (size_t i = 0; i < v->num_dim; i++)
        CU_ASSERT_DOUBLE_EQUAL(dst->data[i], v->data[i], 1e-12);
    fvm_vec_destroyer(dst);
    fvm_vec_destroyer(v);
    return;
}

/* --- norm --- */
void test_vec_norm(void)
{
    double v_data[] = {1.0, 2.0, 3.0};
    double exp_norm = sqrt(1.0*1.0 + 2.0 * 2.0 + 3.0 * 3.0);
    fvm_vector *v = fvm_vec_decl(v_data, 3);

    double r = fvm_vec_norm(v);
    CU_ASSERT_DOUBLE_EQUAL(r, exp_norm, 1e-12);
    fvm_vec_destroyer(v);
    return;
}

/* --- axpy (returns new vector) --- */
void test_vec_axpy(void)
{
    double x_data[] = {1.0, 2.0, 3.0};
    fvm_vector *x = fvm_vec_decl(x_data, 3);

    double a = 2.0;

    double y_data[] = {-2.0, -4.0, -6.0};
    fvm_vector *y = fvm_vec_decl(y_data, 3);

    double exp_res_data[] = {0.0, 0.0, 0.0};
    fvm_vector *exp_res = fvm_vec_decl(exp_res_data, 3);

    fvm_vector *r = fvm_vec_axpy(y, a, x);
    CU_ASSERT_PTR_NOT_NULL(x);
    CU_ASSERT_PTR_NOT_NULL(y);
    CU_ASSERT_PTR_NOT_NULL(r);
    CU_ASSERT_PTR_NOT_NULL(exp_res);

    CU_ASSERT_EQUAL(r->num_dim, exp_res->num_dim);
    for (size_t i = 0; i < r->num_dim; i++)
    CU_ASSERT_DOUBLE_EQUAL(r->data[i], exp_res->data[i], 1e-12);
    fvm_vec_destroyer(r);    
    fvm_vec_destroyer(y);
    fvm_vec_destroyer(x);    
    fvm_vec_destroyer(exp_res);

    return;
}

/* --- axpy in place --- */
void test_vec_axpy_r(void)
{
    double x_data[] = {1.0, 2.0, 3.0};
    fvm_vector *x = fvm_vec_decl(x_data, 3);

    double a = 2.0;

    double y_data[] = {-2.0, -4.0, -6.0};
    fvm_vector *y = fvm_vec_decl(y_data, 3);

    double exp_res_data[] = {0.0, 0.0, 0.0};
    fvm_vector *exp_res = fvm_vec_decl(exp_res_data, 3);

    CU_ASSERT_PTR_NOT_NULL(x);
    CU_ASSERT_PTR_NOT_NULL(y);
    CU_ASSERT_PTR_NOT_NULL(exp_res);

    fvm_vector *y_copy = fvm_vec_copy(y);
    CU_ASSERT_PTR_NOT_NULL(y_copy);
    int ret = fvm_vec_axpy_r(y_copy, a, x);
    CU_ASSERT_EQUAL(ret, 1);
    for (size_t i = 0; i < y_copy->num_dim; i++)
        CU_ASSERT_DOUBLE_EQUAL(y_copy->data[i], exp_res->data[i], 1e-12);
    fvm_vec_destroyer(y_copy);
    fvm_vec_destroyer(y);
    fvm_vec_destroyer(x);    
    fvm_vec_destroyer(exp_res);
    return;
}

/* --- scalar_mul (returns new vector) --- */
void test_vec_scalar_mul()
{
    double v_data[] = {1.0, 2.0, 3.0};
    fvm_vector *v = fvm_vec_decl(v_data, 3);
    
    double val = -2.0;

    double exp_res_data[] = {-2.0, -4.0, -6.0};
    fvm_vector* exp_res = fvm_vec_decl(exp_res_data, 3);

    fvm_vector *r = fvm_vec_scalar_mul(v, val);
    CU_ASSERT_PTR_NOT_NULL(r);
    CU_ASSERT_PTR_NOT_NULL(exp_res);
    CU_ASSERT_PTR_NOT_NULL(v);

    CU_ASSERT_EQUAL(r->num_dim, exp_res->num_dim);
    for (size_t i = 0; i < r->num_dim; i++)
        CU_ASSERT_DOUBLE_EQUAL(r->data[i], exp_res->data[i], 1e-12);
    fvm_vec_destroyer(r);
    fvm_vec_destroyer(v);
    fvm_vec_destroyer(exp_res);
    return;
}

/* --- scalar_mul in place --- */
void test_vec_scalar_mul_r()
{   

    double v_data[] = {1.0, 2.0, 3.0};
    fvm_vector *v = fvm_vec_decl(v_data, 3);
    
    double val = -2.0;

    double exp_res_data[] = {-2.0, -4.0, -6.0};
    fvm_vector* exp_res = fvm_vec_decl(exp_res_data, 3);

    CU_ASSERT_PTR_NOT_NULL(v);
    CU_ASSERT_PTR_NOT_NULL(exp_res);

    fvm_vector *copy = fvm_vec_copy(v);
    CU_ASSERT_PTR_NOT_NULL(copy);
    
    int ret = fvm_vec_scalar_mul_r(copy, val);
    CU_ASSERT_EQUAL(ret, 1);

    for (size_t i = 0; i < copy->num_dim; i++)
        CU_ASSERT_DOUBLE_EQUAL(copy->data[i], exp_res->data[i], 1e-12);
    fvm_vec_destroyer(copy);
    fvm_vec_destroyer(v);
    fvm_vec_destroyer(exp_res);
    return;
}

/* --- scalar_div (returns new vector) --- */
void test_vec_scalar_div(void)
{

    double v_data[] = {1.0, 2.0, 3.0};
    fvm_vector *v = fvm_vec_decl(v_data, 3);
    
    double val = -0.5;

    double exp_res_data[] = {-2.0, -4.0, -6.0};
    fvm_vector* exp_res = fvm_vec_decl(exp_res_data, 3);

    CU_ASSERT_PTR_NOT_NULL(v);
    CU_ASSERT_PTR_NOT_NULL(exp_res);

    fvm_vector *r = fvm_vec_scalar_div(v, val);
    CU_ASSERT_PTR_NOT_NULL(r);
    CU_ASSERT_EQUAL(r->num_dim, exp_res->num_dim);
    for (size_t i = 0; i < r->num_dim; i++)
        CU_ASSERT_DOUBLE_EQUAL(r->data[i], exp_res->data[i], 1e-12);
    fvm_vec_destroyer(r);
    fvm_vec_destroyer(v);
    fvm_vec_destroyer(exp_res);
    return;
}

/* --- scalar_div in place --- */
void test_vec_scalar_div_r(void)
{
    double v_data[] = {1.0, 2.0, 3.0};
    fvm_vector *v = fvm_vec_decl(v_data, 3);
    
    double val = -0.5;

    double exp_res_data[] = {-2.0, -4.0, -6.0};
    fvm_vector* exp_res = fvm_vec_decl(exp_res_data, 3);

    CU_ASSERT_PTR_NOT_NULL(v);
    CU_ASSERT_PTR_NOT_NULL(exp_res);
    
    fvm_vector *copy = fvm_vec_copy(v);
    CU_ASSERT_PTR_NOT_NULL(copy);
    int ret = fvm_vec_scalar_div_r(copy, val);
    CU_ASSERT_EQUAL(ret, 1);
    for (size_t i = 0; i < copy->num_dim; i++)
        CU_ASSERT_DOUBLE_EQUAL(copy->data[i], exp_res->data[i], 1e-12);
    fvm_vec_destroyer(copy);
    fvm_vec_destroyer(v);
    fvm_vec_destroyer(exp_res);
    return;
}

/* --- add_val (returns new vector) --- */
void test_vec_add_val(void)
{
    double v_data[] = {1.0, 2.0, 3.0};
    fvm_vector *v = fvm_vec_decl(v_data, 3);
    
    double val = 2.0;

    double exp_res_data[] = {3.0, 4.0, 5.0};
    fvm_vector* exp_res = fvm_vec_decl(exp_res_data, 3);

    CU_ASSERT_PTR_NOT_NULL(v);
    CU_ASSERT_PTR_NOT_NULL(exp_res);

    fvm_vector *r = fvm_vec_add_val(v, val);
    CU_ASSERT_PTR_NOT_NULL(r);
    CU_ASSERT_EQUAL(r->num_dim, exp_res->num_dim);
    for (size_t i = 0; i < r->num_dim; i++)
        CU_ASSERT_DOUBLE_EQUAL(r->data[i], exp_res->data[i], 1e-12);
    fvm_vec_destroyer(r);
    fvm_vec_destroyer(v);
    fvm_vec_destroyer(exp_res);
    return;
}

/* --- add_val in place --- */
void test_vec_add_val_r(void)
{
    double v_data[] = {1.0, 2.0, 3.0};
    fvm_vector *v = fvm_vec_decl(v_data, 3);
    
    double val = 2.0;

    double exp_res_data[] = {3.0, 4.0, 5.0};
    fvm_vector* exp_res = fvm_vec_decl(exp_res_data, 3);

    CU_ASSERT_PTR_NOT_NULL(v);
    CU_ASSERT_PTR_NOT_NULL(exp_res);

    fvm_vector *copy = fvm_vec_copy(v);
    CU_ASSERT_PTR_NOT_NULL(copy);
    int ret = fvm_vec_add_val_r(copy, val);
    CU_ASSERT_EQUAL(ret, 1);
    for (size_t i = 0; i < copy->num_dim; i++)
        CU_ASSERT_DOUBLE_EQUAL(copy->data[i], exp_res->data[i], 1e-12);
    
    fvm_vec_destroyer(copy);
    fvm_vec_destroyer(v);
    fvm_vec_destroyer(exp_res);
    return;
}

/* --- sub_val (returns new vector) --- */
void test_vec_sub_val(void)
{

    double v_data[] = {1.0, 2.0, 3.0};
    fvm_vector *v = fvm_vec_decl(v_data, 3);
    
    double val = -2.0;

    double exp_res_data[] = {3.0, 4.0, 5.0};
    fvm_vector* exp_res = fvm_vec_decl(exp_res_data, 3);

    CU_ASSERT_PTR_NOT_NULL(v);
    CU_ASSERT_PTR_NOT_NULL(exp_res);

    fvm_vector *r = fvm_vec_sub_val(v, val);
    CU_ASSERT_PTR_NOT_NULL(r);
    CU_ASSERT_EQUAL(r->num_dim, exp_res->num_dim);
    for (size_t i = 0; i < r->num_dim; i++)
        CU_ASSERT_DOUBLE_EQUAL(r->data[i], exp_res->data[i], 1e-12);
    fvm_vec_destroyer(r);
    fvm_vec_destroyer(v);
    fvm_vec_destroyer(exp_res);
    return;
}

/* --- sub_val in place --- */
void test_vec_sub_val_r(void)
{
    double v_data[] = {1.0, 2.0, 3.0};
    fvm_vector *v = fvm_vec_decl(v_data, 3);
    
    double val = -2.0;

    double exp_res_data[] = {3.0, 4.0, 5.0};
    fvm_vector* exp_res = fvm_vec_decl(exp_res_data, 3);

    CU_ASSERT_PTR_NOT_NULL(v);
    CU_ASSERT_PTR_NOT_NULL(exp_res);

    fvm_vector *copy = fvm_vec_copy(v);
    CU_ASSERT_PTR_NOT_NULL(copy);
    int ret = fvm_vec_sub_val_r(copy, val);
    CU_ASSERT_EQUAL(ret, 1);
    for (size_t i = 0; i < copy->num_dim; i++)
        CU_ASSERT_DOUBLE_EQUAL(copy->data[i], exp_res->data[i], 1e-12);
    fvm_vec_destroyer(copy);
    fvm_vec_destroyer(v);
    fvm_vec_destroyer(exp_res);
    return;
}

/* --- sub (returns new vector) --- */
void test_vec_sub(void)
{

    double v_data[] = {1.0, 2.0, 3.0};
    fvm_vector *v = fvm_vec_decl(v_data, 3);

    double u_data[] = {1.0, 2.0, 3.0};
    fvm_vector *u = fvm_vec_decl(u_data, 3);

    double exp_res_data[] = {0.0, 0.0, 0.0};
    fvm_vector* exp_res = fvm_vec_decl(exp_res_data, 3);

    CU_ASSERT_PTR_NOT_NULL(u);
    CU_ASSERT_PTR_NOT_NULL(v);
    CU_ASSERT_PTR_NOT_NULL(exp_res);

    fvm_vector *r = fvm_vec_sub(v, u);
    CU_ASSERT_PTR_NOT_NULL(r);
    CU_ASSERT_EQUAL(r->num_dim, exp_res->num_dim);
    for (size_t i = 0; i < r->num_dim; i++)
        CU_ASSERT_DOUBLE_EQUAL(r->data[i], exp_res->data[i], 1e-12);
    fvm_vec_destroyer(r);
    fvm_vec_destroyer(v);
    fvm_vec_destroyer(u);
    fvm_vec_destroyer(exp_res);
    return;
}

/* --- sub in place --- */
void test_vec_sub_r(void)
{
    double v_data[] = {1.0, 2.0, 3.0};
    fvm_vector *v = fvm_vec_decl(v_data, 3);

    double u_data[] = {1.0, 2.0, 3.0};
    fvm_vector *u = fvm_vec_decl(u_data, 3);

    double exp_res_data[] = {0.0, 0.0, 0.0};
    fvm_vector* exp_res = fvm_vec_decl(exp_res_data, 3);

    CU_ASSERT_PTR_NOT_NULL(v);
    CU_ASSERT_PTR_NOT_NULL(u);
    CU_ASSERT_PTR_NOT_NULL(exp_res);

    fvm_vector *copy = fvm_vec_copy(v);
    CU_ASSERT_PTR_NOT_NULL(copy);
    int ret = fvm_vec_sub_r(copy, u);
    CU_ASSERT_EQUAL(ret, 1);
    for (size_t i = 0; i < copy->num_dim; i++)
        CU_ASSERT_DOUBLE_EQUAL(copy->data[i], exp_res->data[i], 1e-12);
    fvm_vec_destroyer(copy);
    fvm_vec_destroyer(v);
    fvm_vec_destroyer(u);
    fvm_vec_destroyer(exp_res);
    return;
}

/* --- add in place --- */
void test_vec_add_r(void)
{
    double v_data[] = {1.0, 2.0, 3.0};
    fvm_vector *v = fvm_vec_decl(v_data, 3);

    double u_data[] = {-1.0, -2.0, -3.0};
    fvm_vector *u = fvm_vec_decl(u_data, 3);

    double exp_res_data[] = {0.0, 0.0, 0.0};
    fvm_vector* exp_res = fvm_vec_decl(exp_res_data, 3);

    CU_ASSERT_PTR_NOT_NULL(u);
    CU_ASSERT_PTR_NOT_NULL(v);
    CU_ASSERT_PTR_NOT_NULL(exp_res);

    fvm_vector *copy = fvm_vec_copy(v);
    CU_ASSERT_PTR_NOT_NULL(copy);
    int ret = fvm_vec_add_r(copy, u);
    CU_ASSERT_EQUAL(ret, 1);
    for (size_t i = 0; i < copy->num_dim; i++)
        CU_ASSERT_DOUBLE_EQUAL(copy->data[i], exp_res->data[i], 1e-12);
    fvm_vec_destroyer(copy);
    fvm_vec_destroyer(v);
    fvm_vec_destroyer(u);
    fvm_vec_destroyer(exp_res);
    return;    
}

/* --- dot product --- */
void test_vec_dot(void)
{
    double v_data[] = {1.0, 1.0, 1.0};
    fvm_vector *v = fvm_vec_decl(v_data, 3);

    double u_data[] = {1.0, 1.0, - 2.0};
    fvm_vector *u = fvm_vec_decl(u_data, 3);

    double exp_dot = 0.0;

    CU_ASSERT_PTR_NOT_NULL(u);
    CU_ASSERT_PTR_NOT_NULL(v);

    double r = fvm_vec_dot(v, u);
    CU_ASSERT_DOUBLE_EQUAL(r, exp_dot, 1e-12);
    
    fvm_vec_destroyer(v);
    fvm_vec_destroyer(u);
    return;
}

/* --- outer product --- */
void test_vec_outer(void)
{   
    double v_data[] = {1.0, 1.0, 1.0};
    fvm_vector *v = fvm_vec_decl(v_data, 3);

    double u_data[] = {1.0, 1.0, 1.0};
    fvm_vector *u = fvm_vec_decl(u_data, 3);

    double exp_res_data[] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    fvm_mat *exp_res = fvm_mat_decl(exp_res_data, 3, 3);

    fvm_mat *r = fvm_vec_outer(v, u);

    CU_ASSERT_PTR_NOT_NULL(r);
    CU_ASSERT_EQUAL(r->num_rows, exp_res->num_rows);
    CU_ASSERT_EQUAL(r->num_cols, exp_res->num_cols);
    size_t size = (size_t)r->num_rows * r->num_cols;
    for (size_t i = 0; i < size; i++)
        CU_ASSERT_DOUBLE_EQUAL(r->data[i], exp_res->data[i], 1e-12);
    fvm_mat_destroyer(r);
    fvm_mat_destroyer(exp_res);
    fvm_vec_destroyer(u);
    fvm_vec_destroyer(v);
    return;
}

/* --- max / min / mean --- */
void test_vec_max(void)
{
    double v_data[] = {3.0, 10.0, 1.0};
    fvm_vector *v = fvm_vec_decl(v_data, 3);
    double exp_max = 10.0;
    CU_ASSERT_DOUBLE_EQUAL(fvm_vec_max(v), exp_max, 1e-12);
    fvm_vec_destroyer(v);
    return;
}

void test_vec_min(void)
{
    double v_data[] = {3.0, 10.0, 1.0};
    fvm_vector *v = fvm_vec_decl(v_data, 3);
    double exp_min = 1.0;
    CU_ASSERT_DOUBLE_EQUAL(fvm_vec_min(v), exp_min, 1e-12);
    fvm_vec_destroyer(v);
    return;
}

void test_vec_mean(void)
{
    double v_data[] = {3.0, 5.0, 1.0};
    fvm_vector *v = fvm_vec_decl(v_data, 3);
    double exp_mean = 3.0;

    CU_ASSERT_DOUBLE_EQUAL(fvm_vec_mean(v), exp_mean, 1e-12);
    fvm_vec_destroyer(v);
    return;
}

/* --- standard deviation --- */
void test_vec_std_pop(void)
{
    double v_data[] = {3.0, 3.0, 3.0};
    fvm_vector *v = fvm_vec_decl(v_data, 3);
    double exp_std = 0.0;
    CU_ASSERT_DOUBLE_EQUAL(fvm_vec_std_pop(v), exp_std, 1e-10);
    fvm_vec_destroyer(v);
    return;
    
}


/* Initialize test suite */
int init_suite(void) {
    return 0;
}

/* Clean up test suite */
int clean_suite(void) {
    return 0;
}

int main(void)
{

    printf("Starting the test procedure.\n");
    if (CU_initialize_registry() != CUE_SUCCESS) {
        return CU_get_error();
    }

    printf("Registry initialized.\n");

    CU_pSuite pSuite = CU_add_suite("Linalg lib test", init_suite, clean_suite);
    if (pSuite == NULL) {
        CU_cleanup_registry();
        return CU_get_error();
    }
    
    printf("Testing the vector creator.\n");

    if (CU_add_test(pSuite, "Test vector creator", test_vec_creator_destroyer) == NULL ) TEST_ERROR_CLEANUP;
    if (CU_add_test(pSuite, "Test vector copy", test_vec_copy) == NULL) TEST_ERROR_CLEANUP;
    if (CU_add_test(pSuite, "Test vector copy contents", test_vec_copy_contents) == NULL) TEST_ERROR_CLEANUP;
    if (CU_add_test(pSuite, "Test vector norm", test_vec_norm) == NULL) TEST_ERROR_CLEANUP;
    if (CU_add_test(pSuite, "Test vector axpy operation", test_vec_axpy) == NULL) TEST_ERROR_CLEANUP;
    if (CU_add_test(pSuite, "Test vector in-place axpy operation", test_vec_axpy_r) == NULL) TEST_ERROR_CLEANUP;
    if (CU_add_test(pSuite, "Test vector scalar mul", test_vec_scalar_mul) == NULL) TEST_ERROR_CLEANUP;
    if (CU_add_test(pSuite, "Test vector scalar in place mul", test_vec_scalar_mul_r) == NULL) TEST_ERROR_CLEANUP;
    if (CU_add_test(pSuite, "Test vector scalar in place mul", test_vec_scalar_div) == NULL) TEST_ERROR_CLEANUP;
    if (CU_add_test(pSuite, "Test vector scalar in place mul", test_vec_scalar_div_r) == NULL) TEST_ERROR_CLEANUP;
    if (CU_add_test(pSuite, "Test scalar addition to vector", test_vec_add_val) == NULL) TEST_ERROR_CLEANUP;
    if (CU_add_test(pSuite, "Test vector  scalar addition to vector in place", test_vec_add_val_r) == NULL) TEST_ERROR_CLEANUP;
    if (CU_add_test(pSuite, "Test vector  scalar subtraction from vector", test_vec_sub_val) == NULL) TEST_ERROR_CLEANUP;
    if (CU_add_test(pSuite, "Test vector  scalar subtraction from vector in place", test_vec_sub_val_r) == NULL) TEST_ERROR_CLEANUP;
    if (CU_add_test(pSuite, "Test vector subtraction from vector", test_vec_sub) == NULL) TEST_ERROR_CLEANUP;
    if (CU_add_test(pSuite, "Test vector subtraction from vector", test_vec_sub_r) == NULL) TEST_ERROR_CLEANUP;
    if (CU_add_test(pSuite, "Test vector addition to vector", test_vec_add_r) == NULL) TEST_ERROR_CLEANUP;
    if (CU_add_test(pSuite, "Test vector dot product", test_vec_dot) == NULL) TEST_ERROR_CLEANUP;
    if (CU_add_test(pSuite, "Test vector outer (tensor) product", test_vec_outer) == NULL) TEST_ERROR_CLEANUP;
    if (CU_add_test(pSuite, "Test vector max element searching", test_vec_max) == NULL) TEST_ERROR_CLEANUP;
    if (CU_add_test(pSuite, "Test vector min element searching", test_vec_min) == NULL) TEST_ERROR_CLEANUP;
    if (CU_add_test(pSuite, "Test vector element-wise mean", test_vec_mean) == NULL) TEST_ERROR_CLEANUP;
    if (CU_add_test(pSuite, "Test vector element-wise std", test_vec_std_pop) == NULL) TEST_ERROR_CLEANUP;

    printf("Running the creator test...\n");
    CU_basic_set_mode(CU_BRM_VERBOSE);
    CU_basic_run_tests();

    CU_cleanup_registry();
    return CU_get_error();
}