#ifndef FVM_VEC_H
#define FVM_VEC_H

#include "fvm.h"
#include "fvm_mat.h"

/* --- Utility functions --- */
/**
 * @brief Creates an empty vector of given dimension.
 *
 * Allocates a new `fvm_vector` object and a contiguous data buffer of length
 * `num_dim`, initialized to `0.0`.
 *
 * @param num_dim Number of vector elements.
 * @return Pointer to a newly allocated vector, or NULL on error.
 *
 * @note The returned vector must be released with `fvm_vec_destroyer()`.
 */
fvm_vector *fvm_vec_creator(unsigned int num_dim);

fvm_vector *fvm_vec_decl(const double *arr, unsigned int num_dim);
/**
 * @brief Frees a vector and its internal data buffer.
 *
 * Releases the memory owned by `vec`, including its contiguous data array.
 *
 * @param vec Pointer to the vector to destroy.
 *
 * @note It is safe to call this function with NULL.
 */
void fvm_vec_destroyer(fvm_vector *vec);

/**
 * @brief Creates a deep copy of a vector.
 *
 * Allocates a new vector with the same dimension as `v` and copies all
 * elements into it.
 *
 * @param v Pointer to the source vector.
 * @return Pointer to a newly allocated copy, or NULL on error.
 *
 * @note The returned vector must be released with `fvm_vec_destroyer()`.
 */
fvm_vector *fvm_vec_copy(fvm_vector *v);

/**
 * @brief Copies the contents of one vector into another existing vector.
 *
 * Copies all elements from `copy_from` to `copy_to`. Both vectors must have
 * the same dimension.
 *
 * @param copy_to Destination vector.
 * @param copy_from Source vector.
 */
void fvm_vec_copy_contents(fvm_vector *copy_to, fvm_vector *copy_from);

/* --- Mathematical operations ---*/
/**
 * @brief Computes the Euclidean norm of a vector.
 *
 * Returns the 2-norm, defined as the square root of the sum of squared
 * elements.
 *
 * @param v Pointer to the input vector.
 * @return Euclidean norm of `v`, or 0.0 on error.
 */
double fvm_vec_norm(fvm_vector *v);

/**
 * @brief Performs the AXPY operation in place.
 *
 * Updates `y` according to `y = y + a * x`. Both vectors must have the same
 * dimension.
 *
 * @param y Pointer to the vector to modify.
 * @param a Scalar multiplier.
 * @param x Pointer to the input vector.
 * @return 1 on success, 0 on error.
 */
int fvm_vec_axpy_r(fvm_vector *y, double a, fvm_vector *x);

/**
 * @brief Returns the result of the AXPY operation on a copy.
 *
 * Creates a deep copy of `y`, then computes `out = y + a * x`.
 * Both vectors must have the same dimension.
 *
 * @param y Pointer to the first input vector.
 * @param a Scalar multiplier.
 * @param x Pointer to the second input vector.
 * @return Pointer to a newly allocated result vector, or NULL on error.
 *
 * @note Neither input vector is modified.
 * @note The returned vector must be released with `fvm_vec_destroyer()`.
 */
fvm_vector *fvm_vec_axpy(fvm_vector *y, double a, fvm_vector *x);

/**
 * @brief Multiplies a vector by a scalar in place.
 *
 * Scales every element of `v` by `val`.
 *
 * @param v Pointer to the vector to modify.
 * @param val Scalar multiplier.
 * @return 1 on success, 0 on error.
 */
int fvm_vec_scalar_mul_r(fvm_vector *v, double val);
/**
 * @brief Returns a copy of a vector scaled by a scalar.
 *
 * Creates a deep copy of `v` and multiplies every element by `val`.
 *
 * @param v Pointer to the source vector.
 * @param val Scalar multiplier.
 * @return Pointer to a newly allocated scaled vector, or NULL on error.
 *
 * @note The input vector is not modified.
 * @note The returned vector must be released with `fvm_vec_destroyer()`.
 */
fvm_vector *fvm_vec_scalar_mul(fvm_vector *v, double val);

/**
 * @brief Divides a vector by a scalar in place.
 *
 * Divides every element of `v` by `val`.
 *
 * @param v Pointer to the vector to modify.
 * @param val Nonzero divisor.
 * @return 1 on success, 0 on error.
 */
int fvm_vec_scalar_div_r(fvm_vector *v, double val);

/**
 * @brief Returns a copy of a vector divided by a scalar.
 *
 * Creates a deep copy of `v` and divides every element by `val`.
 *
 * @param v Pointer to the source vector.
 * @param val Nonzero divisor.
 * @return Pointer to a newly allocated scaled vector, or NULL on error.
 *
 * @note The input vector is not modified.
 * @note The returned vector must be released with `fvm_vec_destroyer()`.
 */
fvm_vector *fvm_vec_scalar_div(fvm_vector *v, double val);

/**
 * @brief Adds a scalar to every vector entry in place.
 *
 * @param v Pointer to the vector to modify.
 * @param val Scalar value added to each element.
 * @return 1 on success, 0 on error.
 */
int fvm_vec_add_val_r(fvm_vector *v, double val);

/**
 * @brief Returns a copy of a vector with a scalar added to every entry.
 *
 * @param v Pointer to the source vector.
 * @param val Scalar value added to each element.
 * @return Pointer to a newly allocated result vector, or NULL on error.
 *
 * @note The input vector is not modified.
 * @note The returned vector must be released with `fvm_vec_destroyer()`.
 */
fvm_vector *fvm_vec_add_val(fvm_vector *v, double val);

/**
 * @brief Subtracts a scalar from every vector entry in place.
 *
 * @param v Pointer to the vector to modify.
 * @param val Scalar value subtracted from each element.
 * @return 1 on success, 0 on error.
 */
int fvm_vec_sub_val_r(fvm_vector *v, double val);

/**
 * @brief Returns a copy of a vector with a scalar subtracted from every entry.
 *
 * @param v Pointer to the source vector.
 * @param val Scalar value subtracted from each element.
 * @return Pointer to a newly allocated result vector, or NULL on error.
 *
 * @note The input vector is not modified.
 * @note The returned vector must be released with `fvm_vec_destroyer()`.
 */
fvm_vector *fvm_vec_sub_val(fvm_vector *v, double val);

/**
 * @brief Adds one vector to another in place.
 *
 * Updates `v1` according to `v1 = v1 + v2`. Both vectors must have the same
 * dimension.
 *
 * @param v1 Pointer to the vector to modify.
 * @param v2 Pointer to the vector added to `v1`.
 * @return 1 on success, 0 on error.
 */
int fvm_vec_add_r(fvm_vector *v1, fvm_vector *v2);

/**
 * @brief Returns the element-wise sum of two vectors.
 *
 * Creates a deep copy of `v1`, then adds `v2` element-wise.
 *
 * @param v1 Pointer to the first input vector.
 * @param v2 Pointer to the second input vector.
 * @return Pointer to a newly allocated result vector, or NULL on error.
 *
 * @note Neither input vector is modified.
 * @note The returned vector must be released with `fvm_vec_destroyer()`.
 */
fvm_vector *fvm_vec_add(fvm_vector *v1, fvm_vector *v2);

/**
 * @brief Subtracts one vector from another in place.
 *
 * Updates `v1` according to `v1 = v1 - v2`. Both vectors must have the same
 * dimension.
 *
 * @param v1 Pointer to the vector to modify.
 * @param v2 Pointer to the vector subtracted from `v1`.
 * @return 1 on success, 0 on error.
 */
int fvm_vec_sub_r(fvm_vector *v1, fvm_vector *v2);

/**
 * @brief Returns the element-wise difference of two vectors.
 *
 * Creates a deep copy of `v1`, then subtracts `v2` element-wise.
 *
 * @param v1 Pointer to the first input vector.
 * @param v2 Pointer to the second input vector.
 * @return Pointer to a newly allocated result vector, or NULL on error.
 *
 * @note Neither input vector is modified.
 * @note The returned vector must be released with `fvm_vec_destroyer()`.
 */
fvm_vector *fvm_vec_sub(fvm_vector *v1, fvm_vector *v2);

/**
 * @brief Computes the dot product of two vectors.
 *
 * Returns the scalar product `sum_i v1[i] * v2[i]`. Both vectors must have
 * the same dimension.
 *
 * @param v1 Pointer to the first input vector.
 * @param v2 Pointer to the second input vector.
 * @return Dot product of the two vectors, or 0.0 on error.
 */
double fvm_vec_dot(fvm_vector *v1, fvm_vector *v2);

/**
 * @brief Computes the outer product of two vectors.
 *
 * Allocates a matrix of size `v1->num_dim x v2->num_dim` and fills it with
 * `out(i,j) = v1[i] * v2[j]`.
 *
 * @param v1 Pointer to the left-hand vector.
 * @param v2 Pointer to the right-hand vector.
 * @return Pointer to a newly allocated matrix, or NULL on error.
 *
 * @note Neither input vector is modified.
 * @note The returned matrix must be released with `fvm_mat_destroyer()`.
 */
fvm_mat *fvm_vec_outer(fvm_vector *v1, fvm_vector *v2);

/* --- Statistical functions --- */
/**
 * @brief Returns the maximum element of a vector.
 *
 * Scans the vector and returns its largest value.
 *
 * @param v Pointer to the input vector.
 * @return Maximum value in the vector, or 0.0 on error.
 */
double fvm_vec_max(fvm_vector *v);

/**
 * @brief Returns the minimum element of a vector.
 *
 * Scans the vector and returns its smallest value.
 *
 * @param v Pointer to the input vector.
 * @return Minimum value in the vector, or 0.0 on error.
 */
double fvm_vec_min(fvm_vector *v);

/**
 * @brief Returns the arithmetic mean of a vector.
 *
 * Computes the average of all elements in the vector.
 *
 * @param v Pointer to the input vector.
 * @return Mean value of the vector, or 0.0 on error.
 */
double fvm_vec_mean(fvm_vector *v);

/**
 * @brief Returns the population standard deviation of a vector.
 *
 * Computes the population standard deviation of the vector elements using
 * denominator `n`.
 *
 * @param v Pointer to the input vector.
 * @return Population standard deviation, or 0.0 on error.
 */
double fvm_vec_std_pop(fvm_vector *v);

/**
 * @brief Returns the sample standard deviation of a vector.
 *
 * Computes the sample standard deviation of the vector elements using
 * denominator `n - 1`.
 *
 * @param v Pointer to the input vector.
 * @return Sample standard deviation, or 0.0 on error.
 *
 * @note The vector must contain at least two elements.
 */
double fvm_vec_std_samp(fvm_vector *v);
#endif