#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "../include/CFDlinalg/fvm.h"
#include "../include/CFDlinalg/fvm_mat.h"
#include "../include/CFDlinalg/fvm_vec.h"

#define fvm_linalg_solve(...) \
    fvm_linalg_solve_argv((fvm_linalg_solve_args){ \
        .n_iter = 50, \
        .tol = 1e-8, \
        .solver = "jacobi", \
        __VA_ARGS__ \
    })

typedef struct {
    int n_iter;
    double tol;
    const char *solver;
    fvm_linalg_system *system;

} fvm_linalg_solve_args;

typedef struct
{
    unsigned int n_row;
    unsigned int n_col;
    double pivot_val;
} pivot;


/* --- Creator, destroyer, checkers, basic utilities --- */
fvm_linalg_system* fvm_linalg_system_creator(fvm_mat* A, fvm_vector* x, fvm_vector* b, int is_homogenous);
void fvm_linalg_system_destroyer(fvm_linalg_system* system, int destroy_subcomp);
int linalg_system_check(fvm_linalg_system* system);
pivot fvm_linalg_find_pivot(fvm_mat* m, int n_row, int n_col);

/*--- Solver wrappers --- */
int fvm_linalg_solve_argv(fvm_linalg_solve_args args);
int fvm_linalg_solve_b(fvm_linalg_system* system, int n_iter, double tol, const char *solver);

/* --- Mathematical operations ---*/
double residual_inf_norm(fvm_vector* res);
fvm_vector* fvm_mat_vec_prod(fvm_mat* m, fvm_vector* v);
int fvm_linalg_gauss_u(fvm_mat* m);

/* --- SOLVERS --- */
int fvm_linalg_solve_jacobi(fvm_linalg_system* system, int n_iter, double tol);
int fvm_linalg_solve_gauss_seidel(fvm_linalg_system* system, int n_iter, double tol);


fvm_linalg_system* fvm_linalg_system_creator(fvm_mat* A, fvm_vector* x, fvm_vector* b, int is_homogenous)
{
    NULL_CHECK_PTR(A);
    NULL_CHECK_PTR(x);

    if (b == NULL && is_homogenous == 0)
    {
        CFD_ERROR(CFD_NULL_POINTER);
        return NULL;
    }

    if (b != NULL && is_homogenous == 1)
    {
        CFD_ERROR(CFD_SYSTEM_HOMOGENITY_ERROR);
        return NULL;
    }

    if (is_homogenous == 0) {
        DIM_COMPARE_VEC_PTR_NULL(b, x);
    }

    MAT_VEC_DIM_COMPARE_MULT_NULL(A, x);

    fvm_linalg_system* system = malloc(sizeof *system);
    if (system == NULL) {
        CFD_ERROR(CFD_NULL_POINTER);
        return NULL;
    }

    system->A = A;
    system->x = x;

    if (is_homogenous == 1)
    {
        system->is_homogenous = 1;
        system->b = NULL;
    }
    else
    {
        system->is_homogenous = 0;
        system->b = b;
    }

    return system;
}

void fvm_linalg_system_destroyer(fvm_linalg_system* system, int destroy_subcomp)
{
    NULL_CHECK_VOID(system);

    if (destroy_subcomp == 1)
    {
        fvm_mat_destroyer(system->A);
        if (system->is_homogenous == 0)
        {
            fvm_vec_destroyer(system->b);
        }
        
        fvm_vec_destroyer(system->x);

        system->A = NULL;
        system->b = NULL;
        system->x = NULL;
    }
    free(system);
    return;
}

int linalg_system_check(fvm_linalg_system* system)
{

    NULL_CHECK_INT(system);
    NULL_CHECK_INT(system->A);
    NULL_CHECK_INT(system->x);
    MAT_VEC_DIM_COMPARE_MULT_INT(system->A, system->x);

    if (system->b == NULL && system->is_homogenous == 0)
    {
        CFD_ERROR(CFD_NULL_POINTER);
        printf("System should be non-homogenous, but no RHS was provided");
        return 0;
    }

    if (system->b != NULL && system->is_homogenous == 1)
    {
        CFD_ERROR(CFD_SYSTEM_HOMOGENITY_ERROR);
        printf("System should be homogenous, but RHS was provided");
        return 0;
    }

    if (system->is_homogenous == 0) {
        DIM_COMPARE_VEC_PTR_INT(system->b, system->x);
    }

    return 1;
}

pivot fvm_linalg_find_pivot(fvm_mat* m, int n_row, int n_col)
{
    pivot pvt;
    double pivot_abs = fabs(MAT_AT(m, n_row, n_col));
    unsigned int p = n_row;

    for (size_t i = n_row + 1; i < m->num_rows; i++)
    {
        double cand_abs = fabs(MAT_AT(m, i, n_col));
        if (cand_abs > pivot_abs) {
            pivot_abs = cand_abs;
            p = i;
        }
    }

    pvt.n_col = n_col;
    pvt.n_row = p;
    pvt.pivot_val = MAT_AT(m, p, n_col);
    return pvt;
}
/* --- Iterative solvers ---*/

int fvm_linalg_solve_argv(fvm_linalg_solve_args args)
{
    NULL_CHECK_INT(args.system);
    NULL_CHECK_INT(args.system->x);
    NULL_CHECK_INT(args.system->A);
    if (args.system->is_homogenous != 1)
    {
        NULL_CHECK_INT(args.system->b);
    }

    return fvm_linalg_solve_b(args.system, args.n_iter, args.tol, args.solver);
}

int fvm_linalg_solve_b(fvm_linalg_system* system, int n_iter, double tol, const char *solver)
{
    if (linalg_system_check(system) == 0)
    {
        printf("Linear system check failed!\n");
        return 0;
    }

    if (strcmp(solver, "jacobi") == 0)
    {
        return fvm_linalg_solve_jacobi(system, n_iter, tol);
    }
    if (strcmp(solver, "gauss-seidel") == 0)
    {
        return fvm_linalg_solve_gauss_seidel(system, n_iter, tol);
    }

    printf("Unknown solver: %s\n", solver);
    return 0;

}
/* --- Mathematical operations ---*/

double residual_inf_norm(fvm_vector* res)
{
    NULL_CHECK_INT(res);

    double max_abs = 0.0;
    for (size_t i = 0; i < res->num_dim; ++i) {
        double a = fabs(res->data[i]);
        if (a > max_abs) max_abs = a;
    }
    return max_abs;
}

fvm_vector* fvm_mat_vec_prod(fvm_mat* m, fvm_vector* v)
{
    NULL_CHECK_INT(m);
    NULL_CHECK_INT(v);
    MAT_VEC_DIM_COMPARE_MULT_INT(m, v);

    fvm_vector* res = fvm_vec_creator(m->num_rows);
    NULL_CHECK_INT(res);

    for (size_t i = 0; i < m->num_rows; i++) {
        double sum = 0.0;
        double* row = m->data + (size_t)i * m->num_cols;

        for (size_t j = 0; j < m->num_cols; j++) {
            sum += row[j] * v->data[j];
        }

        res->data[i] = sum;
    }

    return res;
}

int fvm_linalg_gauss_u(fvm_mat* m)
{
    NULL_CHECK_INT(m);

    if (m->num_rows != m->num_cols) {
        CFD_ERROR(CFD_INVALID_DIMENSIONS);
        printf("Gaussian elimination requires a square matrix\n");
        return 0;
    }

    unsigned int n_rows = m->num_rows;
    unsigned int n_cols = m->num_cols;

    for (size_t i = 0; i < n_rows; i++)
    {
        pivot pvt = fvm_linalg_find_pivot(m, i, i);

        if (fabs(pvt.pivot_val) == 0.0) {
            CFD_ERROR(CFD_INVALID_DIMENSIONS);
            printf("Gaussian elimination failed: zero pivot at column %zu\n", i);
            return 0;
        }

        fvm_mat_swap_rows_r(m, i, pvt.n_row);

        for (size_t k = i + 1; k < n_rows; k++)
        {
            double mltplr = MAT_AT(m, k, i) / MAT_AT(m, i, i);

            for (size_t j = i; j < n_cols; j++)
            {
                MAT_AT(m, k, j) -= mltplr * MAT_AT(m, i, j);
            }
        }
    }

    return 1;
}

/* --- SOLVERS --- */

int fvm_linalg_solve_jacobi(fvm_linalg_system* system, int n_iter, double tol)
{
    NULL_CHECK_INT(system);
    NULL_CHECK_INT(system->A);
    NULL_CHECK_INT(system->x);

    if (system->is_homogenous != 1) {
        NULL_CHECK_INT(system->b);
    }

    int dim_x = (int)system->x->num_dim;
    int dim_A = (int)system->A->num_cols;

    if (system->A->num_rows != system->A->num_cols) {
        CFD_ERROR(CFD_INVALID_DIMENSIONS);
        printf("Jacobi requires a square matrix\n");
        return 0;
    }

    if (dim_x != dim_A) {
        CFD_ERROR(CFD_INVALID_DIMENSIONS);
        printf("Jacobi requires compatible dimensions between A and x\n");
        return 0;
    }

    fvm_vector* x_old = fvm_vec_copy(system->x);
    fvm_vector* x_new = fvm_vec_copy(system->x);

    if (x_old == NULL || x_new == NULL) {
        CFD_ERROR(CFD_NULL_POINTER);
        fvm_vec_destroyer(x_old);
        fvm_vec_destroyer(x_new);
        return 0;
    }

    for (int t = 0; t < n_iter; ++t)
    {
        for (int i = 0; i < dim_x; ++i)
        {
            double aii = MAT_AT(system->A, i, i);
            if (aii == 0.0) {
                CFD_ERROR(CFD_INVALID_DIMENSIONS);
                printf("Jacobi failed: zero diagonal entry at row %d\n", i);
                fvm_vec_destroyer(x_old);
                fvm_vec_destroyer(x_new);
                return 0;
            }

            double sigma = 0.0;
            for (int j = 0; j < dim_A; ++j)
            {
                if (j == i) continue;
                sigma += MAT_AT(system->A, i, j) * MAT_V(x_old, j);
            }

            if (system->is_homogenous == 1) {
                MAT_V(x_new, i) = -sigma / aii;
            } else {
                MAT_V(x_new, i) = (MAT_V(system->b, i) - sigma) / aii;
            }
        }

        fvm_vector* res = fvm_mat_vec_prod(system->A, x_new);
        if (res == NULL) {
            CFD_ERROR(CFD_NULL_POINTER);
            fvm_vec_destroyer(x_old);
            fvm_vec_destroyer(x_new);
            return 0;
        }

        if (system->is_homogenous != 1) {
            if (fvm_vec_sub_r(res, system->b) == 0) {
                fvm_vec_destroyer(res);
                fvm_vec_destroyer(x_old);
                fvm_vec_destroyer(x_new);
                return 0;
            }
        }

        double res_norm = residual_inf_norm(res);
        printf("Jacobi iter %d, residual inf norm = %.12e\n", t + 1, res_norm);

        if (res_norm < tol)
        {
            fvm_vec_copy_contents(system->x, x_new);
            fvm_vec_destroyer(res);
            fvm_vec_destroyer(x_old);
            fvm_vec_destroyer(x_new);
            return 1;
        }

        fvm_vec_copy_contents(x_old, x_new);
        fvm_vec_destroyer(res);
    }

    fvm_vec_copy_contents(system->x, x_new);
    fvm_vec_destroyer(x_old);
    fvm_vec_destroyer(x_new);

    printf("Jacobi method did not converge within %d iterations\n", n_iter);
    return 0;
}

int fvm_linalg_solve_gauss_seidel(fvm_linalg_system* system, int n_iter, double tol)
{
    NULL_CHECK_INT(system);
    NULL_CHECK_INT(system->A);
    NULL_CHECK_INT(system->x);

    if (system->is_homogenous != 1) {
        NULL_CHECK_INT(system->b);
    }

    int dim_x = (int)system->x->num_dim;
    int dim_A = (int)system->A->num_cols;

    if (system->A->num_rows != system->A->num_cols) {
        CFD_ERROR(CFD_INVALID_DIMENSIONS);
        printf("Gauss-Seidel requires a square matrix\n");
        return 0;
    }

    if (dim_x != dim_A) {
        CFD_ERROR(CFD_INVALID_DIMENSIONS);
        printf("Gauss-Seidel requires compatible dimensions between A and x\n");
        return 0;
    }

    fvm_vector* x_old = fvm_vec_copy(system->x);
    fvm_vector* x_new = fvm_vec_copy(system->x);

    if (x_old == NULL || x_new == NULL) {
        CFD_ERROR(CFD_NULL_POINTER);
        fvm_vec_destroyer(x_old);
        fvm_vec_destroyer(x_new);
        return 0;
    }

    for (int t = 0; t < n_iter; ++t)
    {
        fvm_vec_copy_contents(x_new, x_old);

        for (int i = 0; i < dim_x; ++i)
        {
            double aii = MAT_AT(system->A, i, i);
            if (aii == 0.0) {
                CFD_ERROR(CFD_INVALID_DIMENSIONS);
                printf("Gauss-Seidel failed: zero diagonal entry at row %d\n", i);
                fvm_vec_destroyer(x_old);
                fvm_vec_destroyer(x_new);
                return 0;
            }

            double sigma = 0.0;

            for (int j = 0; j < dim_A; ++j)
            {
                if (j == i) continue;

                if (j < i) {
                    sigma += MAT_AT(system->A, i, j) * MAT_V(x_new, j);
                } else {
                    sigma += MAT_AT(system->A, i, j) * MAT_V(x_old, j);
                }
            }

            if (system->is_homogenous == 1) {
                MAT_V(x_new, i) = -sigma / aii;
            } else {
                MAT_V(x_new, i) = (MAT_V(system->b, i) - sigma) / aii;
            }
        }

        fvm_vector* res = fvm_mat_vec_prod(system->A, x_new);
        if (res == NULL) {
            CFD_ERROR(CFD_NULL_POINTER);
            fvm_vec_destroyer(x_old);
            fvm_vec_destroyer(x_new);
            return 0;
        }

        if (system->is_homogenous != 1) {
            if (fvm_vec_sub_r(res, system->b) == 0) {
                fvm_vec_destroyer(res);
                fvm_vec_destroyer(x_old);
                fvm_vec_destroyer(x_new);
                return 0;
            }
        }

        double res_norm = residual_inf_norm(res);
        printf("Gauss-Seidel iter %d, residual inf norm = %.12e\n", t + 1, res_norm);

        if (res_norm < tol)
        {
            fvm_vec_copy_contents(system->x, x_new);
            fvm_vec_destroyer(res);
            fvm_vec_destroyer(x_old);
            fvm_vec_destroyer(x_new);
            return 1;
        }

        fvm_vec_copy_contents(x_old, x_new);
        fvm_vec_destroyer(res);
    }

    fvm_vec_copy_contents(system->x, x_new);
    fvm_vec_destroyer(x_old);
    fvm_vec_destroyer(x_new);

    printf("Gauss-Seidel method did not converge within %d iterations\n", n_iter);
    return 0;
}