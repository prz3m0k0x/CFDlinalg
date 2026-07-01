#ifndef FVM_MAT_H
#define FVM_MAT_H

#include "fvm.h"
#include "fvm_vec.h"

/* --- Creator, destroyer, checkers, basic utilities --- */

/** 
    *@brief Function creates an empty matrix object with given number of rows and colums.
    *
    *    @param num_rows - number of rows of a matrix,
    *    @param num_cols - number of columns of a matrix,
    *Returns: 
    *    - a fvm_mat* object representing matrix with given dimension 
    *
    *The fvm_mat object is a struct with three entries:
    *    ->num_rows - number of rows of a matrix,
    *    ->num_cols - number of columns of a matrix,
    *    ->data - pointer to heap allocated memory block with size num_rows * num_cols * sizeof(double)
    *Important notes:
    *    @note rows and columns are conceptual - matrix data is a single code block,
    *    @note data is initialized with 0.0 in a format of double
    */
fvm_mat *fvm_mat_creator(unsigned int num_rows, unsigned int num_cols);

/**
 * @brief Create a matrix and initialize its contents from a flat array.
 *
 * Allocates a new fvm_mat object and a contiguous data buffer sized
 * num_rows * num_cols doubles, then copies the values from the caller-supplied
 * array arr into the newly allocated matrix storage.
 *
 * The input array arr is interpreted in row-major order: element at row r,
 * column c is arr[r * num_cols + c]. The created matrix owns its storage and
 * must be released with fvm_mat_destroyer().
 *
 * Parameters
 * ----------
 * @param arr        Pointer to a contiguous array of double values to copy
 *                   from. This function does not modify arr.
 * @param num_rows   Number of rows of the matrix to create (must be > 0).
 * @param num_cols   Number of columns of the matrix to create (must be > 0).
 *
 * Returns
 * -------
 * @return fvm_mat * On success, a pointer to a newly allocated fvm_mat whose
 *                   ->data contains a copy of the contents of arr.
 * @return NULL      On error: if arr is NULL, if num_rows or num_cols is zero,
 *                   or if memory allocation fails. In case of error, no memory
 *                   is leaked and a diagnostic is reported via CFD_ERROR.
 *
 * Ownership & lifetime
 * --------------------
 * - The returned fvm_mat owns the memory in its ->data field and must be
 *   freed with fvm_mat_destroyer() when no longer needed.
 * - The caller retains ownership of arr; this function only reads from it and
 *   does not store the pointer.
 *
 * Notes & requirements
 * --------------------
 * - arr must point to at least num_rows * num_cols consecutive doubles.
 * - The function assumes row-major layout for both arr and the internal
 *   fvm_mat->data layout.
 * - The function is safe to call in test code to create matrices from static
 *   initializer arrays, e.g. `double A[] = { ... }; fvm_mat *m = fvm_mat_decl(A, r, c);
 */
fvm_mat *fvm_mat_decl(const double *arr, unsigned int num_rows, unsigned int num_cols);

/**
 * @brief Frees a matrix object and its internal data buffer.
 *
 * Destroys a heap-allocated `fvm_mat` instance by first releasing the
 * contiguous data block stored in `m->data`, then releasing the matrix
 * structure itself.
 *
 * The function is safe to call with a NULL pointer: in that case it returns
 * immediately without performing any operation.
 *
 * @param m Pointer to a matrix object previously created by `fvm_mat_creator`
 *          or another function allocating a compatible `fvm_mat`.
 *
 * @note After this call, the pointer becomes invalid and must not be used.
 * @note This function assumes `m->data` either points to heap memory allocated
 *       by the library or is NULL.
 */
void fvm_mat_destroyer(fvm_mat *m);
/**
 * @brief Counts matrix entries whose absolute value exceeds a tolerance.
 *
 * Scans all elements of the matrix and returns the number of entries for
 * which `fabs(value) > tol`. This is useful for estimating sparsity or
 * detecting numerically significant values after arithmetic operations.
 *
 * @param m Pointer to the matrix to inspect.
 * @param tol Nonnegative tolerance used as the zero threshold.
 * @return Number of elements with absolute value greater than `tol`.
 *
 * @note Values with `fabs(value) <= tol` are treated as zero.
 * @note If the count exceeds `INT_MAX`, the function reports an error and
 *       returns 0.
 * @note `tol` should be nonnegative.
 */
int fvm_mat_nonzero_count(fvm_mat *m, double tol);
/**
 * @brief Creates a matrix filled with random values from a given interval.
 *
 * Allocates a new `fvm_mat` object with dimensions `num_rows x num_cols`
 * and fills its contiguous data array with random values generated in the
 * interval `[min, max]` using `fvm_rand_intvl(min, max)`.
 *
 * @param num_rows Number of matrix rows.
 * @param num_cols Number of matrix columns.
 * @param min Lower bound of the random interval.
 * @param max Upper bound of the random interval.
 * @return Pointer to a newly allocated random matrix, or NULL if allocation
 *         fails or the dimensions are invalid.
 *
 * @note The returned matrix must be released with `fvm_mat_destroyer()`.
 * @note This function reseeds the pseudo-random number generator using
 *       `time(NULL)` on each call.
 * @warning Repeated calls within the same second may generate identical
 *          matrices because the generator is reseeded every time.
 */
fvm_mat *fvm_mat_rand(unsigned int num_rows, unsigned int num_cols, double min, double max);
/**
 * @brief Creates a square matrix with dimension num x num.
 *
 * @param num Number of rows and columns.
 * @return Pointer to a newly allocated square matrix, or NULL on error.
 *
 * @note The returned matrix must be released with fvm_mat_destroyer().
 */
fvm_mat *fvm_mat_square(unsigned int num);
/**
 * @brief Creates a square matrix filled with random values.
 *
 * Allocates a `num x num` matrix and fills it with random values from the
 * interval `[min, max]` using `fvm_rand_intvl(min, max)`.
 *
 * @param num Matrix dimension.
 * @param min Lower bound of the random interval.
 * @param max Upper bound of the random interval.
 * @return Pointer to a newly allocated random square matrix, or NULL on error.
 *
 * @note The returned matrix must be released with `fvm_mat_destroyer()`.
 * @note If `min > max`, the function reports `CFD_OUT_OF_RANGE` and returns NULL.
 * @warning Repeated calls within the same second may generate identical
 *          matrices if the RNG is reseeded elsewhere per call.
 */
fvm_mat *fvm_mat_square_rand(unsigned int num, double min, double max);
/**
 * @brief Creates an identity matrix.
 *
 * Allocates a `num x num` square matrix and sets the diagonal entries to 1,
 * with all off-diagonal entries initialized to 0.
 *
 * @param num Matrix dimension.
 * @return Pointer to a newly allocated identity matrix, or NULL on error.
 *
 * @note The returned matrix must be released with `fvm_mat_destroyer()`.
 */
fvm_mat *fvm_mat_eye(unsigned int num);
/**
 * @brief Creates a matrix filled with ones.
 *
 * Allocates a `num_rows x num_cols` matrix and initializes every entry to 1.0.
 *
 * @param num_rows Number of rows.
 * @param num_cols Number of columns.
 * @return Pointer to a newly allocated matrix filled with ones, or NULL on error.
 *
 * @note The returned matrix must be released with fvm_mat_destroyer().
 */
fvm_mat *fvm_mat_ones(unsigned int num_rows, unsigned int num_cols);
/**
 * @brief Creates a matrix filled with a constant value.
 *
 * Allocates a `num_rows x num_cols` matrix and initializes every entry to `a`.
 *
 * @param num_rows Number of rows.
 * @param num_cols Number of columns.
 * @param a Value used to fill every matrix entry.
 * @return Pointer to a newly allocated constant-filled matrix, or NULL on error.
 *
 * @note The returned matrix must be released with fvm_mat_destroyer().
 */
fvm_mat *fvm_mat_full(unsigned int num_rows, unsigned int num_cols, double a);
/**
 * @brief Creates a diagonal matrix.
 *
 * Allocates a `num_rows x num_cols` matrix, initializes all entries to `0.0`,
 * and sets the diagonal entries to `val` up to `min(num_rows, num_cols)`.
 *
 * @param num_rows Number of rows.
 * @param num_cols Number of columns.
 * @param val Value placed on the main diagonal.
 * @return Pointer to a newly allocated diagonal matrix, or NULL on error.
 *
 * @note The returned matrix must be released with fvm_mat_destroyer().
 */
fvm_mat *fvm_mat_diag(unsigned int num_rows, unsigned int num_cols, double val);
/**
 * @brief Creates a deep copy of a matrix.
 *
 * Allocates a new matrix with the same dimensions as `m` and copies all
 * elements into the new object.
 *
 * @param m Pointer to the source matrix.
 * @return Pointer to a newly allocated copy, or NULL on error.
 *
 * @note The returned matrix must be released with fvm_mat_destroyer().
 */
fvm_mat *fvm_mat_copy(fvm_mat *m);

/* --- Shape, size and matrix alternatiion and manipulation functions --- */
/**
 * @brief Returns a copy of the matrix with one row removed.
 *
 * Allocates a new matrix containing all rows of `m` except row `row_n`.
 * Rows below `row_n` are shifted up by one.
 *
 * @param m Pointer to the source matrix.
 * @param row_n Index of the row to remove.
 * @return Pointer to a new matrix with one fewer row, or NULL on error.
 *
 * @note The returned matrix must be released with fvm_mat_destroyer().
 */
fvm_mat *fvm_mat_remove_row(fvm_mat *m, unsigned int row_n);
/**
 * @brief Returns a copy of the matrix with one column removed.
 *
 * Allocates a new matrix containing all columns of `m` except column `col_n`.
 * Columns to the right of `col_n` are shifted left by one.
 *
 * @param m Pointer to the source matrix.
 * @param col_n Index of the column to remove.
 * @return Pointer to a new matrix with one fewer column, or NULL on error.
 *
 * @note The input matrix is not modified.
 * @note The returned matrix must be released with `fvm_mat_destroyer()`.
 */
fvm_mat *fvm_mat_remove_col(fvm_mat *m, unsigned int col_n);
/**
 * @brief Swaps two rows of a matrix in place.
 *
 * Exchanges rows `r1` and `r2` directly in the input matrix.
 *
 * @param m Pointer to the matrix to modify.
 * @param r1 Index of the first row.
 * @param r2 Index of the second row.
 * @return 1 on success, 0 on error.
 */
int fvm_mat_swap_rows_r(fvm_mat *m, unsigned int r1, unsigned int r2);
/**
 * @brief Returns a copy of the matrix with two rows swapped.
 *
 * Creates a deep copy of `m` and exchanges rows `r1` and `r2` in the copy.
 *
 * @param m Pointer to the source matrix.
 * @param r1 Index of the first row.
 * @param r2 Index of the second row.
 * @return Pointer to a new matrix with swapped rows, or NULL on error.
 *
 * @note The input matrix is not modified.
 * @note The returned matrix must be released with `fvm_mat_destroyer()`.
 */
fvm_mat *fvm_mat_swap_rows(fvm_mat *m, unsigned int r1, unsigned int r2);
/**
 * @brief Swaps two columns of a matrix in place.
 *
 * Exchanges columns `c1` and `c2` directly in the input matrix.
 *
 * @param m Pointer to the matrix to modify.
 * @param c1 Index of the first column.
 * @param c2 Index of the second column.
 * @return 1 on success, 0 on error.
 */
int fvm_mat_swap_cols_r(fvm_mat *m, unsigned int c1, unsigned int c2);
/**
 * @brief Returns a copy of the matrix with two columns swapped.
 *
 * Creates a deep copy of `m` and exchanges columns `c1` and `c2` in the copy.
 *
 * @param m Pointer to the source matrix.
 * @param c1 Index of the first column.
 * @param c2 Index of the second column.
 * @return Pointer to a new matrix with swapped columns, or NULL on error.
 *
 * @note The input matrix is not modified.
 * @note The returned matrix must be released with `fvm_mat_destroyer()`.
 */
fvm_mat *fvm_mat_swap_cols(fvm_mat *m, unsigned int c1, unsigned int c2);
/**
 * @brief Returns the transpose of a matrix.
 *
 * Allocates a new matrix of size `m->num_cols x m->num_rows` and copies
 * elements so that `out(j,i) = m(i,j)`.
 *
 * @param m Pointer to the source matrix.
 * @return Pointer to a newly allocated transposed matrix, or NULL on error.
 *
 * @note The input matrix is not modified.
 * @note The returned matrix must be released with `fvm_mat_destroyer()`.
 */
fvm_mat *fvm_mat_transpose(fvm_mat *m);

/* --- Accesing functions ---*/
/**
 * @brief Returns a copy of one matrix row as a 1 x N matrix.
 *
 * Allocates a new matrix containing the values from row `row_n`.
 * The returned matrix has shape `1 x m->num_cols`.
 *
 * @param m Pointer to the source matrix.
 * @param row_n Index of the row to extract.
 * @return Pointer to a newly allocated 1-row matrix, or NULL on error.
 *
 * @note The input matrix is not modified.
 * @note The returned matrix must be released with `fvm_mat_destroyer()`.
 */
fvm_mat *fvm_mat_rows_get(fvm_mat *m, unsigned int row_n);

/**
 * @brief Returns a copy of one matrix column as an M x 1 matrix.
 *
 * Allocates a new matrix containing the values from column `col_n`.
 * The returned matrix has shape `m->num_rows x 1`.
 *
 * @param m Pointer to the source matrix.
 * @param col_n Index of the column to extract.
 * @return Pointer to a newly allocated 1-column matrix, or NULL on error.
 *
 * @note The input matrix is not modified.
 * @note The returned matrix must be released with `fvm_mat_destroyer()`.
 */
fvm_mat *fvm_mat_col_get(fvm_mat *m, unsigned int col_n);

/* --- Matrix algebra ---- */

/**
 * @brief Multiplies a matrix row by a scalar in place.
 *
 * Scales every element of row `row_n` by `val` directly in `m`.
 *
 * @param m Pointer to the matrix to modify.
 * @param row_n Index of the row to scale.
 * @param val Scalar multiplier.
 * @return 1 on success, 0 on error.
 */
int fvm_mat_row_mul_r(fvm_mat *m, unsigned int row_n, double val);

/**
 * @brief Returns a copy of the matrix with one row scaled by a scalar.
 *
 * Creates a deep copy of `m` and multiplies every element of row `row_n`
 * by `val` in the copy.
 *
 * @param m Pointer to the source matrix.
 * @param row_n Index of the row to scale.
 * @param val Scalar multiplier.
 * @return Pointer to a new matrix, or NULL on error.
 *
 * @note The input matrix is not modified.
 * @note The returned matrix must be released with `fvm_mat_destroyer()`.
 */
fvm_mat *fvm_mat_row_mul(fvm_mat *m, unsigned int row_n, double val);

/**
 * @brief Multiplies a matrix column by a scalar in place.
 *
 * Scales every element of column `col_n` by `val` directly in `m`.
 *
 * @param m Pointer to the matrix to modify.
 * @param col_n Index of the column to scale.
 * @param val Scalar multiplier.
 * @return 1 on success, 0 on error.
 */
int fvm_mat_col_mul_r(fvm_mat *m, unsigned int col_n, double val);

/**
 * @brief Returns a copy of the matrix with one column scaled by a scalar.
 *
 * Creates a deep copy of `m` and multiplies every element of column `col_n`
 * by `val` in the copy.
 *
 * @param m Pointer to the source matrix.
 * @param col_n Index of the column to scale.
 * @param val Scalar multiplier.
 * @return Pointer to a new matrix, or NULL on error.
 *
 * @note The input matrix is not modified.
 * @note The returned matrix must be released with `fvm_mat_destroyer()`.
 */
fvm_mat *fvm_mat_col_mul(fvm_mat *m, unsigned int col_n, double val);

/**
 * @brief Divides a matrix row by a scalar in place.
 *
 * Scales every element of row `n_row` by `1/a` directly in `m`.
 *
 * @param m Pointer to the matrix to modify.
 * @param n_row Index of the row to divide.
 * @param a Nonzero divisor.
 * @return 1 on success, 0 on error.
 */
int fvm_mat_row_div_r(fvm_mat *m, int n_row, double a);

/**
 * @brief Returns a copy of the matrix with one row divided by a scalar.
 *
 * Creates a deep copy of `m` and divides every element of row `n_row`
 * by `a` in the copy.
 *
 * @param m Pointer to the source matrix.
 * @param n_row Index of the row to divide.
 * @param a Nonzero divisor.
 * @return Pointer to a new matrix, or NULL on error.
 *
 * @note The input matrix is not modified.
 * @note The returned matrix must be released with `fvm_mat_destroyer()`.
 */
fvm_mat *fvm_mat_row_div(fvm_mat *m, int n_row, double a);
/**
 * @brief Divides a matrix column by a scalar in place.
 *
 * Divides every element of column `col_n` by `val` directly in `m`.
 *
 * @param m Pointer to the matrix to modify.
 * @param col_n Index of the column to divide.
 * @param val Nonzero divisor.
 * @return 1 on success, 0 on error.
 */
int fvm_mat_col_div_r(fvm_mat *m, unsigned int col_n, double val);

/**
 * @brief Returns a copy of the matrix with one column divided by a scalar.
 *
 * Creates a deep copy of `m` and divides every element of column `col_n`
 * by `val` in the copy.
 *
 * @param m Pointer to the source matrix.
 * @param col_n Index of the column to divide.
 * @param val Nonzero divisor.
 * @return Pointer to a new matrix, or NULL on error.
 *
 * @note The input matrix is not modified.
 * @note The returned matrix must be released with `fvm_mat_destroyer()`.
 */
fvm_mat *fvm_mat_col_div(fvm_mat *m, unsigned int col_n, double val);

/**
 * @brief Adds one matrix to another in place.
 *
 * Adds the elements of `m2` to `m1` element-wise and stores the result in `m1`.
 * Both matrices must have the same dimensions.
 *
 * @param m1 Pointer to the matrix to modify.
 * @param m2 Pointer to the matrix added to `m1`.
 * @return 1 on success, 0 on error.
 */
int fvm_mat_add_r(fvm_mat *m1, fvm_mat *m2);

/**
 * @brief Returns the element-wise sum of two matrices.
 *
 * Creates a deep copy of `m1`, then adds the elements of `m2` to the copy.
 * Both matrices must have the same dimensions.
 *
 * @param m1 Pointer to the first input matrix.
 * @param m2 Pointer to the second input matrix.
 * @return Pointer to a new matrix containing the sum, or NULL on error.
 *
 * @note Neither input matrix is modified.
 * @note The returned matrix must be released with `fvm_mat_destroyer()`.
 */
fvm_mat *fvm_mat_add(fvm_mat *m1, fvm_mat *m2);

/**
 * @brief Multiplies a matrix by a scalar in place.
 *
 * Scales every element of `m` by `val` directly in the input matrix.
 *
 * @param m Pointer to the matrix to modify.
 * @param val Scalar multiplier.
 * @return 1 on success, 0 on error.
 */
int fvm_mat_scalar_mul_r(fvm_mat *m, double val);

/**
 * @brief Returns a copy of the matrix scaled by a scalar.
 *
 * Creates a deep copy of `m` and multiplies every element by `val`.
 *
 * @param m Pointer to the source matrix.
 * @param val Scalar multiplier.
 * @return Pointer to a new scaled matrix, or NULL on error.
 *
 * @note The input matrix is not modified.
 * @note The returned matrix must be released with `fvm_mat_destroyer()`.
 */
fvm_mat *fvm_mat_scalar_mul(fvm_mat *m, double val);

/**
 * @brief Computes the matrix product of two matrices.
 *
 * Allocates a new matrix and computes the standard matrix product
 * `m1 * m2`. The number of columns of `m1` must equal the number of rows
 * of `m2`.
 *
 * @param m1 Pointer to the left-hand matrix.
 * @param m2 Pointer to the right-hand matrix.
 * @return Pointer to a new matrix containing the product, or NULL on error.
 *
 * @note Neither input matrix is modified.
 * @note The returned matrix must be released with `fvm_mat_destroyer()`.
 */
fvm_mat *fvm_mat_prod_naive(fvm_mat *m1, fvm_mat *m2);

/**
 * @brief Computes the matrix product using a row-wise accumulation algorithm.
 *
 * Allocates a new matrix containing the product `m1 * m2`.
 * The number of columns of `m1` must equal the number of rows of `m2`.
 * Much preffered over naive method.
 * @param m1 Pointer to the left-hand matrix.
 * @param m2 Pointer to the right-hand matrix.
 * @return Pointer to a newly allocated matrix containing the product,
 *         or NULL on error.
 *
 * @note Neither input matrix is modified.
 * @note The returned matrix must be released with `fvm_mat_destroyer()`.
 */
fvm_mat *fvm_mat_prod_cubicle_soft_6(fvm_mat *m1, fvm_mat *m2);

/* --- Miscelanious functions --- */
double fvm_rand_intrvl(double min, double max);


#endif