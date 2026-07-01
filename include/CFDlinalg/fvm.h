#ifndef FVM_FVM_H
#define FVM_FVM_H

/* --- Error handling macros and functions */
typedef enum
{
    CFD_OK = 0,
    CFD_INVALID_DIMENSIONS,
    CFD_ALLOCATION_FAILED,
    CFD_NULL_POINTER,
    CFD_OUT_OF_RANGE,
    CFD_NONEQUAL_DIM,
    CFD_SYSTEM_HOMOGENITY_ERROR,
    CFD_DIVISION_BY_ZERO
} cfd_error_t;

static inline const char *fvm_error_string(cfd_error_t err)
{
    switch (err)
    {
    case CFD_OK:
        return "OK";
    case CFD_INVALID_DIMENSIONS:
        return "invalid dimensions";
    case CFD_ALLOCATION_FAILED:
        return "allocation failed";
    case CFD_NULL_POINTER:
        return "null pointer";
    case CFD_OUT_OF_RANGE:
        return "out of range";
    case CFD_NONEQUAL_DIM:
        return "matricies have wrong dimensions to be cast";
    case CFD_SYSTEM_HOMOGENITY_ERROR:
        return "system should be homogenous, but b vector was provided";
    case CFD_DIVISION_BY_ZERO:
        return "Division by zero occured";
    default:
        return "unknown error";
    }
}

#define CFD_ERROR(err, ...)                                   \
    do                                                        \
    {                                                         \
        fprintf(stderr, "[CFD_ERROR] %s:%d: %s",              \
                __FILE__, __LINE__, fvm_error_string((err))); \
        if (sizeof(#__VA_ARGS__) > 1)                         \
            fprintf(stderr, ": " __VA_ARGS__);                \
        fprintf(stderr, "\n");                                \
    } while (0)

#define MAT_AT(A, i, j) ((A)->data[(size_t)(i) * (A)->num_cols + (size_t)(j)])
#define MAT_V(v, i) ((v)->data[(size_t)(i)])

/*--- Default structures shared by other files ---*/

typedef struct
{
    unsigned int num_dim;
    double *data;
} fvm_vector;

typedef struct
{
    unsigned int num_rows;
    unsigned int num_cols;
    double *data;
    int is_square;
} fvm_mat;

typedef struct
{
    unsigned int num_rows;
    unsigned int num_cols;
    unsigned int n_nonzero;
    double *data;
    unsigned int *col_ind;
    unsigned int *indptr;
} fvm_csm_mat;

typedef struct
{
    fvm_mat *A;
    fvm_vector *x;
    fvm_vector *b;
    int is_homogenous;
} fvm_linalg_system;

/* --- Miscelanious macro checks for dimensions, null pointers, boundness and many more for common function return type ---*/
#define DIV_0_CHECK_PTR(a)                   \
    do                                       \
    {                                        \
        if ((a) < 1e-08)                     \
        {                                    \
            CFD_ERROR(CFD_DIVISION_BY_ZERO); \
            return NULL;                     \
        }                                    \
    } while (0);

#define DIV_0_CHECK_INT(a)                   \
    do                                       \
    {                                        \
        if ((a) < 1e-08)                     \
        {                                    \
            CFD_ERROR(CFD_DIVISION_BY_ZERO); \
            return 0;                        \
        }                                    \
    } while (0);

#define DIV_0_CHECK_VOID(a)                  \
    do                                       \
    {                                        \
        if ((a) < 1e-08)                     \
        {                                    \
            CFD_ERROR(CFD_DIVISION_BY_ZERO); \
            return;                          \
        }                                    \
    } while (0);

#define DIV_0_CHECK_DBL(a)                   \
    do                                       \
    {                                        \
        if ((a) < 1e-08)                     \
        {                                    \
            CFD_ERROR(CFD_DIVISION_BY_ZERO); \
            return 0.0;                      \
        }                                    \
    } while (0);

#define DIM_CHECK_IS_POS_PTR(v)                \
    do                                         \
    {                                          \
        if (((v)->num_dim) <= 0)               \
        {                                      \
            CFD_ERROR(CFD_INVALID_DIMENSIONS); \
            return NULL;                       \
        }                                      \
    } while (0)

#define DIM_CHECK_IS_POS_INT(v)                \
    do                                         \
    {                                          \
        if (((v)->num_dim) <= 0)               \
        {                                      \
            CFD_ERROR(CFD_INVALID_DIMENSIONS); \
            return 0;                          \
        }                                      \
    } while (0)

#define DIM_CHECK_IS_POS_VOID(v)               \
    do                                         \
    {                                          \
        if (((v)->num_dim) <= 0)               \
        {                                      \
            CFD_ERROR(CFD_INVALID_DIMENSIONS); \
            return;                            \
        }                                      \
    } while (0)

#define DIM_CHECK_IS_POS_DBL(v)                \
    do                                         \
    {                                          \
        if (((v)->num_dim) <= 0)               \
        {                                      \
            CFD_ERROR(CFD_INVALID_DIMENSIONS); \
            return 0.0;                        \
        }                                      \
    } while (0)

#define NULL_CHECK_PTR(v)                \
    do                                   \
    {                                    \
        if ((v) == NULL)                 \
        {                                \
            CFD_ERROR(CFD_NULL_POINTER); \
            return NULL;                 \
        }                                \
    } while (0)

#define NULL_CHECK_INT(v)                \
    do                                   \
    {                                    \
        if ((v) == NULL)                 \
        {                                \
            CFD_ERROR(CFD_NULL_POINTER); \
            return 0;                    \
        }                                \
    } while (0)

#define NULL_CHECK_VOID(v)               \
    do                                   \
    {                                    \
        if ((v) == NULL)                 \
        {                                \
            CFD_ERROR(CFD_NULL_POINTER); \
            return;                      \
        }                                \
    } while (0)

#define NULL_CHECK_DBL(v)                \
    do                                   \
    {                                    \
        if ((v) == NULL)                 \
        {                                \
            CFD_ERROR(CFD_NULL_POINTER); \
            return 0.0;                  \
        }                                \
    } while (0)

#define DIM_COMPARE_VEC_PTR_NULL(v, w)         \
    do                                         \
    {                                          \
        if ((v)->num_dim != (w)->num_dim)      \
        {                                      \
            CFD_ERROR(CFD_INVALID_DIMENSIONS); \
            return NULL;                       \
        }                                      \
    } while (0)

#define DIM_COMPARE_VEC_PTR_INT(v, w)          \
    do                                         \
    {                                          \
        if ((v)->num_dim != (w)->num_dim)      \
        {                                      \
            CFD_ERROR(CFD_INVALID_DIMENSIONS); \
            return 0;                          \
        }                                      \
    } while (0)

#define DIM_COMPARE_VEC_PTR_VOID(v, w)         \
    do                                         \
    {                                          \
        if ((v)->num_dim != (w)->num_dim)      \
        {                                      \
            CFD_ERROR(CFD_INVALID_DIMENSIONS); \
            return;                            \
        }                                      \
    } while (0)

#define MAT_DIM_COMPARE_ADD_NULL(A, B)                                        \
    do                                                                        \
    {                                                                         \
        if ((A)->num_rows != (B)->num_rows || (A)->num_cols != (B)->num_cols) \
        {                                                                     \
            CFD_ERROR(CFD_INVALID_DIMENSIONS);                                \
            return NULL;                                                      \
        }                                                                     \
    } while (0)

#define MAT_DIM_COMPARE_ADD_INT(A, B)                                         \
    do                                                                        \
    {                                                                         \
        if ((A)->num_rows != (B)->num_rows || (A)->num_cols != (B)->num_cols) \
        {                                                                     \
            CFD_ERROR(CFD_INVALID_DIMENSIONS);                                \
            return 0;                                                         \
        }                                                                     \
    } while (0)

#define MAT_DIM_COMPARE_ADD_VOID(A, B)                                        \
    do                                                                        \
    {                                                                         \
        if ((A)->num_rows != (B)->num_rows || (A)->num_cols != (B)->num_cols) \
        {                                                                     \
            CFD_ERROR(CFD_INVALID_DIMENSIONS);                                \
            return;                                                           \
        }                                                                     \
    } while (0)


#define MAT_DIM_COMPARE_MULT_INT(A, B)         \
    do                                         \
    {                                          \
        if ((A)->num_cols != (B)->num_rows)    \
        {                                      \
            CFD_ERROR(CFD_INVALID_DIMENSIONS); \
            return 0;                          \
        }                                      \
    } while (0)

#define MAT_DIM_COMPARE_MULT_VOID(A, B)        \
    do                                         \
    {                                          \
        if ((A)->num_cols != (B)->num_rows)    \
        {                                      \
            CFD_ERROR(CFD_INVALID_DIMENSIONS); \
            return;                            \
        }                                      \
    } while (0)

#define MAT_VEC_DIM_COMPARE_MULT_NULL(A, v)    \
    do                                         \
    {                                          \
        if ((A)->num_cols != (v)->num_dim)     \
        {                                      \
            CFD_ERROR(CFD_INVALID_DIMENSIONS); \
            return NULL;                       \
        }                                      \
    } while (0)

#define MAT_VEC_DIM_COMPARE_MULT_INT(A, v)     \
    do                                         \
    {                                          \
        if ((A)->num_cols != (v)->num_dim)     \
        {                                      \
            CFD_ERROR(CFD_INVALID_DIMENSIONS); \
            return 0;                          \
        }                                      \
    } while (0)

#define DIM_COMPARE_VEC_PTR_DBL(v, w)          \
    do                                         \
    {                                          \
        if ((v)->num_dim != (w)->num_dim)      \
        {                                      \
            CFD_ERROR(CFD_INVALID_DIMENSIONS); \
            return 0.0;                        \
        }                                      \
    } while (0)

#define MAT_VEC_DIM_COMPARE_MULT_VOID(A, v)    \
    do                                         \
    {                                          \
        if ((A)->num_cols != (v)->num_dim)     \
        {                                      \
            CFD_ERROR(CFD_INVALID_DIMENSIONS); \
            return;                            \
        }                                      \
    } while (0)

#define MAT_DIM_COMPARE_MULT_NULL(m1, m2)     \
    do                                        \
    {                                         \
        if ((m1)->num_cols != (m2)->num_rows) \
        {                                     \
            CFD_ERROR(CFD_NONEQUAL_DIM);      \
            return NULL;                      \
        }                                     \
    } while (0)

#define MAT_COL_RANGE_CHECK_PTR(m, c)    \
    do                                   \
    {                                    \
        if ((c) >= (m)->num_cols)        \
        {                                \
            CFD_ERROR(CFD_OUT_OF_RANGE); \
            return NULL;                 \
        }                                \
    } while (0)

#define MAT_COL_RANGE_CHECK_INT(m, c)    \
    do                                   \
    {                                    \
        if ((c) >= (m)->num_cols)        \
        {                                \
            CFD_ERROR(CFD_OUT_OF_RANGE); \
            return 0;                    \
        }                                \
    } while (0)

#define MAT_ROW_RANGE_CHECK_PTR(m, r)    \
    do                                   \
    {                                    \
        if ((r) >= (m)->num_rows)        \
        {                                \
            CFD_ERROR(CFD_OUT_OF_RANGE); \
            return NULL;                 \
        }                                \
    } while (0)

#define MAT_ROW_RANGE_CHECK_INT(m, r)    \
    do                                   \
    {                                    \
        if ((r) >= (m)->num_rows)        \
        {                                \
            CFD_ERROR(CFD_OUT_OF_RANGE); \
            return 0;                    \
        }                                \
    } while (0)

#define MAT_2ROW_RANGE_CHECK_PTR(m, r1, r2)                 \
    do                                                      \
    {                                                       \
        if ((r1) >= (m)->num_rows || (r2) >= (m)->num_rows) \
        {                                                   \
            CFD_ERROR(CFD_OUT_OF_RANGE);                    \
            return NULL;                                    \
        }                                                   \
    } while (0)

#define MAT_2ROW_RANGE_CHECK_INT(m, r1, r2)                 \
    do                                                      \
    {                                                       \
        if ((r1) >= (m)->num_rows || (r2) >= (m)->num_rows) \
        {                                                   \
            CFD_ERROR(CFD_OUT_OF_RANGE);                    \
            return 0;                                       \
        }                                                   \
    } while (0)

#define MAT_2COL_RANGE_CHECK_PTR(m, c1, c2)                 \
    do                                                      \
    {                                                       \
        if ((c1) >= (m)->num_cols || (c2) >= (m)->num_cols) \
        {                                                   \
            CFD_ERROR(CFD_OUT_OF_RANGE);                    \
            return NULL;                                    \
        }                                                   \
    } while (0)

#define MAT_2COL_RANGE_CHECK_INT(m, c1, c2)                 \
    do                                                      \
    {                                                       \
        if ((c1) >= (m)->num_cols || (c2) >= (m)->num_cols) \
        {                                                   \
            CFD_ERROR(CFD_OUT_OF_RANGE);                    \
            return 0;                                       \
        }                                                   \
    } while (0)

#endif
