#include <stdlib.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "fvm.h"

/*Utilities functions*/

fvm_vector *fvm_vec_creator(unsigned int num_dim)
{
    if (num_dim == 0) {
        CFD_ERROR(CFD_INVALID_DIMENSIONS);
        return NULL;
    }

    fvm_vector *vec = calloc(1, sizeof *vec);
    if (vec == NULL) {
        CFD_ERROR(CFD_ALLOCATION_FAILED);
        return NULL;
    }

    vec->num_dim = num_dim;
    vec->data = calloc((size_t)num_dim, sizeof *vec->data);
    if (vec->data == NULL) {
        free(vec);
        CFD_ERROR(CFD_ALLOCATION_FAILED);
    return NULL;
    }
}

void fvm_vec_destroyer(fvm_vector *vec)
{
    NULL_CHECK_VOID(vec);
    free(vec->data);
    free(vec);
}

fvm_vector *fvm_vec_copy(fvm_vector *v)
{
    NULL_CHECK_PTR(v);
    NULL_CHECK_PTR(v->data);

    fvm_vector *out = fvm_vec_creator(v->num_dim);
    NULL_CHECK_PTR(out);

    for (size_t i = 0; i < v->num_dim; i++) {
        MAT_V(out, i) = MAT_V(v, i);
    }

    return out;
}

void fvm_vec_copy_contents(fvm_vector *copy_to, fvm_vector *copy_from)
{
    NULL_CHECK_VOID(copy_to);
    NULL_CHECK_VOID(copy_to->data);
    NULL_CHECK_VOID(copy_from);
    NULL_CHECK_VOID(copy_from->data);

    DIM_COMPARE_VEC_PTR_VOID(copy_to, copy_from);

    for (size_t i = 0; i < copy_from->num_dim; i++) {
        MAT_V(copy_to, i) = MAT_V(copy_from, i);
    }
}

fvm_vector *fvm_vec_decl(const double *arr, unsigned int num_dim)
{
    if (arr == NULL || num_dim == 0) {
        CFD_ERROR(CFD_INVALID_DIMENSIONS);
        return NULL;
    }

    fvm_vector *vec = fvm_vec_creator(num_dim);
    if (vec == NULL) {
        return NULL;
    }

    for (unsigned int i = 0; i < num_dim; ++i) {
        vec->data[i] = arr[i];
    }

    return vec;
}

/*Mathematical operations*/

double fvm_vec_norm(fvm_vector *v)
{
    NULL_CHECK_DBL(v);
    NULL_CHECK_DBL(v->data);

    double sum = 0.0;
    for (size_t i = 0; i < v->num_dim; i++) {
        double x = MAT_V(v, i);
        sum += x * x;
    }

    return sqrt(sum);
}

int fvm_vec_axpy_r(fvm_vector *y, double a, fvm_vector *x)
{
    NULL_CHECK_INT(x);
    NULL_CHECK_INT(y);
    DIM_COMPARE_VEC_PTR_INT(x, y);

    for (size_t i = 0; i < y->num_dim; i++) {
        MAT_V(y, i) += a * MAT_V(x, i);
    }

    return 1;
}

fvm_vector *fvm_vec_axpy(fvm_vector *y, double a, fvm_vector *x)
{
    NULL_CHECK_PTR(x);
    NULL_CHECK_PTR(y);
    DIM_COMPARE_VEC_PTR_NULL(x, y);

    fvm_vector *out = fvm_vec_copy(y);
    NULL_CHECK_PTR(out);

    for (size_t i = 0; i < out->num_dim; i++) {
        MAT_V(out, i) += a * MAT_V(x, i);
    }

    return out;
}

int fvm_vec_scalar_mul_r(fvm_vector *v, double val)
{
    NULL_CHECK_INT(v);

    for (size_t i = 0; i < v->num_dim; i++) {
        MAT_V(v, i) *= val;
    }

    return 1;
}

fvm_vector *fvm_vec_scalar_mul(fvm_vector *v, double val)
{
    NULL_CHECK_PTR(v);

    fvm_vector *out = fvm_vec_copy(v);
    NULL_CHECK_PTR(out);

    for (size_t i = 0; i < out->num_dim; i++) {
        MAT_V(out, i) *= val;
    }

    return out;
}

int fvm_vec_scalar_div_r(fvm_vector *v, double val)
{
    NULL_CHECK_INT(v);

    if (val == 0.0) {
        CFD_ERROR(CFD_DIVISION_BY_ZERO);
        return 0;
    }

    for (size_t i = 0; i < v->num_dim; i++) {
        MAT_V(v, i) /= val;
    }

    return 1;
}

fvm_vector *fvm_vec_scalar_div(fvm_vector *v, double val)
{
    NULL_CHECK_PTR(v);

    if (val == 0.0) {
        CFD_ERROR(CFD_DIVISION_BY_ZERO);
        return NULL;
    }

    fvm_vector *out = fvm_vec_copy(v);
    NULL_CHECK_PTR(out);

    for (size_t i = 0; i < out->num_dim; i++) {
        MAT_V(out, i) /= val;
    }

    return out;
}

int fvm_vec_add_val_r(fvm_vector *v, double val)
{
    NULL_CHECK_INT(v);

    for (size_t i = 0; i < v->num_dim; i++) {
        MAT_V(v, i) += val;
    }

    return 1;
}

fvm_vector *fvm_vec_add_val(fvm_vector *v, double val)
{
    NULL_CHECK_PTR(v);

    fvm_vector *out = fvm_vec_copy(v);
    NULL_CHECK_PTR(out);

    for (size_t i = 0; i < out->num_dim; i++) {
        MAT_V(out, i) += val;
    }

    return out;
}

int fvm_vec_sub_val_r(fvm_vector *v, double val)
{
    NULL_CHECK_INT(v);

    for (size_t i = 0; i < v->num_dim; i++) {
        MAT_V(v, i) -= val;
    }

    return 1;
}

fvm_vector *fvm_vec_sub_val(fvm_vector *v, double val)
{
    NULL_CHECK_PTR(v);

    fvm_vector *out = fvm_vec_copy(v);
    NULL_CHECK_PTR(out);

    for (size_t i = 0; i < out->num_dim; i++) {
        MAT_V(out, i) -= val;
    }

    return out;
}

int fvm_vec_add_r(fvm_vector *v1, fvm_vector *v2)
{
    NULL_CHECK_INT(v1);
    NULL_CHECK_INT(v2);
    DIM_COMPARE_VEC_PTR_INT(v1, v2);

    for (size_t i = 0; i < v1->num_dim; i++) {
        MAT_V(v1, i) += MAT_V(v2, i);
    }

    return 1;
}

fvm_vector *fvm_vec_add(fvm_vector *v1, fvm_vector *v2)
{
    NULL_CHECK_PTR(v1);
    NULL_CHECK_PTR(v2);
    DIM_COMPARE_VEC_PTR_NULL(v1, v2);

    fvm_vector *out = fvm_vec_copy(v1);
    NULL_CHECK_PTR(out);

    for (size_t i = 0; i < out->num_dim; i++) {
        MAT_V(out, i) += MAT_V(v2, i);
    }

    return out;
}

int fvm_vec_sub_r(fvm_vector *v1, fvm_vector *v2)
{
    NULL_CHECK_INT(v1);
    NULL_CHECK_INT(v2);
    DIM_COMPARE_VEC_PTR_INT(v1, v2);

    for (size_t i = 0; i < v1->num_dim; i++) {
        MAT_V(v1, i) -= MAT_V(v2, i);
    }

    return 1;
}

fvm_vector *fvm_vec_sub(fvm_vector *v1, fvm_vector *v2)
{
    NULL_CHECK_PTR(v1);
    NULL_CHECK_PTR(v2);
    DIM_COMPARE_VEC_PTR_NULL(v1, v2);

    fvm_vector *out = fvm_vec_copy(v1);
    NULL_CHECK_PTR(out);

    for (size_t i = 0; i < out->num_dim; i++) {
        MAT_V(out, i) -= MAT_V(v2, i);
    }

    return out;
}

double fvm_vec_dot(fvm_vector *v1, fvm_vector *v2)
{
    NULL_CHECK_DBL(v1);
    NULL_CHECK_DBL(v2);
    DIM_COMPARE_VEC_PTR_DBL(v1, v2);

    double sum = 0.0;
    for (size_t i = 0; i < v1->num_dim; i++) {
        sum += MAT_V(v1, i) * MAT_V(v2, i);
    }

    return sum;
}

fvm_mat *fvm_vec_outer(fvm_vector *v1, fvm_vector *v2)
{
    NULL_CHECK_PTR(v1);
    NULL_CHECK_PTR(v2);

    fvm_mat *out = fvm_mat_creator(v1->num_dim, v2->num_dim);
    NULL_CHECK_PTR(out);

    for (size_t i = 0; i < v1->num_dim; i++) {
        for (size_t j = 0; j < v2->num_dim; j++) {
            MAT_AT(out, i, j) = MAT_V(v1, i) * MAT_V(v2, j);
        }
    }

    return out;
}

/*Statistical functions*/
double fvm_vec_max(fvm_vector *v)
{
    NULL_CHECK_DBL(v);
    NULL_CHECK_DBL(v->data);

    DIM_CHECK_IS_POS_DBL(v);

    double mx = MAT_V(v, 0);
    for (size_t i = 1; i < v->num_dim; i++) {
        if (MAT_V(v, i) > mx) {
            mx = MAT_V(v, i);
        }
    }

    return mx;
}

double fvm_vec_min(fvm_vector *v)
{
    NULL_CHECK_DBL(v);

    DIM_CHECK_IS_POS_DBL(v);

    double mn = MAT_V(v, 0);
    for (size_t i = 1; i < v->num_dim; i++) {
        if (MAT_V(v, i) < mn) {
            mn = MAT_V(v, i);
        }
    }

    return mn;
}

double fvm_vec_mean(fvm_vector *v)
{
    NULL_CHECK_DBL(v);

    DIM_CHECK_IS_POS_DBL(v);

    double mean = MAT_V(v, 0);
    for (size_t i = 1; i < v->num_dim; i++) {
        mean += MAT_V(v, i);
    }
    mean /= v->num_dim;
    return mean;
}

double fvm_vec_std_pop(fvm_vector *v)
{
    NULL_CHECK_DBL(v);
    NULL_CHECK_DBL(v->data);
    DIM_CHECK_IS_POS_DBL(v);

    double mean = fvm_vec_mean(v);
    double sum = 0.0;

    for (size_t i = 0; i < v->num_dim; i++) {
        double d = MAT_V(v, i) - mean;
        sum += d * d;
    }

    return sqrt(sum / v->num_dim);
}


double fvm_vec_std_samp(fvm_vector *v)
{
    NULL_CHECK_DBL(v);
    NULL_CHECK_DBL(v->data);

    if (v->num_dim < 2) {
        CFD_ERROR(CFD_INVALID_DIMENSIONS);
        return 0.0;
    }

    double mean = fvm_vec_mean(v);
    double sum = 0.0;

    for (size_t i = 0; i < v->num_dim; i++) {
        double d = MAT_V(v, i) - mean;
        sum += d * d;
    }

    return sqrt(sum / (v->num_dim - 1));
}