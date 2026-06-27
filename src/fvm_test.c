#include "../include/CFDlinalg/fvm.h"
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

int main()
{

    /* v0: zero vector, 3x1 */
    double v0_data[] = {0.0, 0.0, 0.0};
    fvm_vector *v0 = fvm_vec_decl(v0_data, 3);

    /* v1: simple positive vector, 3x1 */
    double v1_data[] = {1.0, 2.0, 3.0};
    fvm_vector *v1 = fvm_vec_decl(v1_data, 3);

    /* v2: mixed signs, 3x1 */
    double v2_data[] = {-1.0, 4.0, -2.0};
    fvm_vector *v2 = fvm_vec_decl(v2_data, 3);

    /* v3: unit basis vector e1, 3x1 */
    double v3_data[] = {1.0, 0.0, 0.0};
    fvm_vector *v3 = fvm_vec_decl(v3_data, 3);

    /* v4: RHS for 4x4 solver tests */
    double v4_data[] = {15.0, 10.0, 10.0, 10.0};
    fvm_vector *v4 = fvm_vec_decl(v4_data, 4);
    /* A0: 3x3 zero matrix */
    double A0_data[] = {
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0
    };
    fvm_mat* A0 = fvm_mat_decl(A0_data, 3, 3);

    /* A1: 3x3 identity matrix */
    double A1_data[] = {
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0
    };
    fvm_mat* A1 = fvm_mat_decl(A1_data, 3, 3);

    /* A2: 3x3 diagonal matrix */
    double A2_data[] = {
        2.0, 0.0, 0.0,
        0.0, 5.0, 0.0,
        0.0, 0.0, -3.0
    };
    fvm_mat* A2 = fvm_mat_decl(A2_data, 3, 3);

    /* A3: 3x3 general dense matrix */
    double A3_data[] = {
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
        7.0, 8.0, 10.0
    };
    fvm_mat* A3 = fvm_mat_decl(A3_data, 3, 3);

    /* A4: 3x3 singular matrix */
    double A4_data[] = {
        1.0, 2.0, 3.0,
        2.0, 4.0, 6.0,
        1.0, 1.0, 1.0
    };
    fvm_mat* A4 = fvm_mat_decl(A4_data, 3, 3);

    /* A5: 3x3 symmetric positive definite matrix */
    double A5_data[] = {
        4.0, 1.0, 1.0,
        1.0, 3.0, 0.0,
        1.0, 0.0, 2.0
    };
    fvm_mat* A5 = fvm_mat_decl(A5_data, 3, 3);

    /* A6: 4x4 strictly diagonally dominant matrix */
    double A6_data[] = {
        10.0, -1.0,  2.0,  0.0,
        -1.0, 11.0, -1.0,  3.0,
        2.0, -1.0, 10.0, -1.0,
        0.0,  3.0, -1.0,  8.0
    };
    fvm_mat* A6 = fvm_mat_decl(A6_data, 4, 4);

    /* R1: 2x3 rectangular matrix */
    double R1_data[] = {
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0
    };
    fvm_mat* R1 = fvm_mat_decl(R1_data, 2, 3);

    /* R2: 3x2 rectangular matrix */
    double R2_data[] = {
        1.0, 2.0,
        3.0, 4.0,
        5.0, 6.0
    };
    fvm_mat* R2 = fvm_mat_decl(R2_data, 3, 2);
}

void test_vec_add(v, w)
{
    fvm_vector *r = fvm_vec_add(v, w);
    
    CU_ASSERT_PTR_NOT_NULL(r);
    CU_ASSERT_EQUAL(r->num_dim, 3);

    CU_ASSERT_DOUBLE_EQUAL(r->data[0], 5.0, 1e-12);
    CU_ASSERT_DOUBLE_EQUAL(r->data[1], 7.0, 1e-12);
    CU_ASSERT_DOUBLE_EQUAL(r->data[2], 9.0, 1e-12);

    fvm_vec_destroyer(r);
}