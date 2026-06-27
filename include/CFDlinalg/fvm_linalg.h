#ifndef FVM_LINALG_H
#define FVM_LINALG_H

#include "fvm.h"
#include "fvm_vec.h"
#include "fvm_mat.h"

#define fvm_linalg_solve(...) \
    fvm_linalg_solve_argv((fvm_linalg_solve_args){ \
        .n_iter = 50, \
        .tol = 1e-8, \
        .solver = "jacobi", \
        __VA_ARGS__ \
    })


typedef struct pivot {
    unsigned int n_row;
    unsigned int n_col;
    double pivot_val;
} pivot;

typedef struct fvm_linalg_solve_args {
    int n_iter;
    double tol;
    const char *solver;
    fvm_linalg_system *system;
} fvm_linalg_solve_args;

/* --- Creator, destroyer, checkers, basic utilities --- */
fvm_linalg_system* fvm_linalg_system_creator(fvm_mat* A, fvm_vector* x, fvm_vector* b, int is_homogenous);
void fvm_linalg_system_destroyer(fvm_linalg_system* system, int destroy_subcomp);
int linalg_system_check(fvm_linalg_system* system);
pivot fvm_linalg_find_pivot(fvm_mat* m, int n_row, int n_col);

/*--- Solver wrappers --- */
int fvm_linalg_solve_argv(fvm_linalg_solve_args args);
int fvm_linalg_solve_b(fvm_linalg_system* system, int n_iter, double tol, const char *solver);

/* --- Mathematical operations ---*/
double max_scaled_residual(fvm_vector* res);
fvm_vector* fvm_mat_vec_prod(fvm_mat* m, fvm_vector* v);
int fvm_linalg_gauss_u(fvm_mat* m);

/* --- SOLVERS --- */
int fvm_linalg_solve_jacobi(fvm_linalg_system* system, int n_iter, double tol);
int fvm_linalg_solve_gauss_seidel(fvm_linalg_system* system, int n_iter, double tol);

#endif