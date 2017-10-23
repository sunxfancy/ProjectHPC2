#include <stdlib.h>
#include <stdio.h>
#include <time.h>

double*
createMatrix(unsigned int n, unsigned int m) {
    return (double*) calloc(sizeof(double), n*m+1);
}

double*
createMatrixWithRandomData(unsigned int n, unsigned int m) {
    double* data; unsigned int i;
    data = createMatrix(n, m);
    for (i = 0; i < n*m; ++i) {
        data[i] = (double)(rand() % 9987) / 100.0;
    }
    return data;
}

int
checkEqual(double *a, double *b, unsigned int n, unsigned int m) {
    unsigned int i;
    for (i = 0; i< n*m; ++i) {
        if (a[i] != b[i]) {
            printf("index: i = %d, j = %d\n", i / m, i % m);
            return 0;
        }
    }
    return 1;
}

void timespec_diff(struct timespec *start, struct timespec *stop,
                   struct timespec *result)
{
    if ((stop->tv_nsec - start->tv_nsec) < 0) {
        result->tv_sec = stop->tv_sec - start->tv_sec - 1;
        result->tv_nsec = stop->tv_nsec - start->tv_nsec + 1000000000;
    } else {
        result->tv_sec = stop->tv_sec - start->tv_sec;
        result->tv_nsec = stop->tv_nsec - start->tv_nsec;
    }
    return;
}

void printMatrix(double* c, unsigned int n, unsigned int m, unsigned int lc) {
    unsigned int i, j;
    for (i = 0; i < n; ++i) {
        for (j = 0; j < m; ++j)
            printf("%.2f ", c[i*lc+j]);
        printf("\n");
    }
}

inline void printMatrix(double* c, unsigned int n, unsigned int m) {
    printMatrix(c, n, m, m);
}

typedef void (*matrixMutiply)(double *a, double *b, double *c, unsigned int n, unsigned int n1, unsigned int n2);

double* runTest(matrixMutiply func, double *a, double *b, unsigned int n, unsigned int n1, unsigned int n2) {
    double *c, timed, n3, p;
    struct timespec ts1,ts2,diff;
    c = createMatrix(n, n2);
    clock_gettime(CLOCK_REALTIME, &ts1);
    func(a,b,c,n,n1,n2);
    clock_gettime(CLOCK_REALTIME, &ts2);
    timespec_diff(&ts1,&ts2,&diff);
    timed = (double)(diff.tv_sec) + 1e-9 * (double)(diff.tv_nsec);
    n3 = (double)n;
    p = 2*n3*n3*n3/timed*1e-9;
    printf("performace: %lf Gflops\n", p);
    printf("timespec is %lu s %lu ns\n", diff.tv_sec, diff.tv_nsec);
    return c;
}

void
dgemm_ikj(double *a, double *b, double *c, unsigned int n, unsigned int n1, unsigned int n2) {
    unsigned int i,j,k;
    for (i=0; i<n; i++)
        for (k=0; k<n1; k++) {
            register double r = a[i*n1+k];
            for (j=0; j<n2; j++)
                c[i*n2+j] += r * b[k*n2+j];
        }
}


void
dgemm_mixed(double *a, double *b, double *c, unsigned int n, unsigned int n1, unsigned int n2, unsigned int la, unsigned int lb, unsigned int lc) {
    const unsigned int B = 30;
    unsigned int i, j, k, i1, j1, k1, m, m1, m2, r, p;
    /* ikj – blocked version algorithm*/
    m = n/B*B; m1 = n1/B*B; m2 = n2/B*B;
    p = (B/3)*3;
    r = B - 2;
    for (i = 0; i < m; i+=B)
        for (k = 0; k < m1; k+=B) 
            for (j = 0; j < m2; j+=B) 
                /* B x B mini matrix multiplications */
                {
                    for (i1 = i; i1 < i+r; i1+=3)
                        for (j1 = j; j1 < j+r; j1+=3) {
                            unsigned int t = i1*lc+j1; unsigned int t1 = t+lc; unsigned int t2 = t+2*lc;
                            register double c00 = c[t];  register double c01 = c[t+1];  register double c02 = c[t+2];
                            register double c10 = c[t1]; register double c11 = c[t1+1]; register double c12 = c[t1+2];
                            register double c20 = c[t2]; register double c21 = c[t2+1]; register double c22 = c[t2+2];
                            for (k1 = k; k1 < k+r; k1+=3) {
                                unsigned int ta = i1*la+k1; unsigned int ta1 = ta+la; unsigned int ta2 = ta+2*la;
                                unsigned int tb = k1*lb+j1; unsigned int tb1 = tb+lb; unsigned int tb2 = tb+2*lb;
                
                                register double a00 = a[ta]; register double a10 = a[ta1];  register double a20 = a[ta2];
                                register double b00 = b[tb]; register double b01 = b[tb+1]; register double b02 = b[tb+2];
                                c00 += a00*b00; c01 += a00*b01; c02 += a00*b02;
                                c10 += a10*b00; c11 += a10*b01; c12 += a10*b02;
                                c20 += a20*b00; c21 += a20*b01; c22 += a20*b02;
                
                                a00 = a[ta+1]; a10 = a[ta1+1]; a20 = a[ta2+1];
                                b00 = b[tb1];  b01 = b[tb1+1]; b02 = b[tb1+2];
                                c00 += a00*b00; c01 += a00*b01; c02 += a00*b02;
                                c10 += a10*b00; c11 += a10*b01; c12 += a10*b02;
                                c20 += a20*b00; c21 += a20*b01; c22 += a20*b02;
                
                                a00 = a[ta+2]; a10 = a[ta1+2]; a20 = a[ta2+2];
                                b00 = b[tb2];  b01 = b[tb2+1]; b02 = b[tb2+2];
                                c00 += a00*b00; c01 += a00*b01; c02 += a00*b02;
                                c10 += a10*b00; c11 += a10*b01; c12 += a10*b02;
                                c20 += a20*b00; c21 += a20*b01; c22 += a20*b02;
                            }
                
                            for (k1 = k+p; k1 < k+B; ++k1) {
                                unsigned int ta = i1*la+k1; unsigned int ta1 = ta+la; unsigned int ta2 = ta+2*la;
                                unsigned int tb = k1*lb+j1;
                                register double a00 = a[ta]; register double a10 = a[ta1];  register double a20 = a[ta2];
                                register double b00 = b[tb]; register double b01 = b[tb+1]; register double b02 = b[tb+2];
                                c00 += a00*b00; c01 += a00*b01; c02 += a00*b02;
                                c10 += a10*b00; c11 += a10*b01; c12 += a10*b02;
                                c20 += a20*b00; c21 += a20*b01; c22 += a20*b02;
                            }
                
                            c[t]  = c00; c[t+1]  = c01; c[t+2]  = c02;
                            c[t1] = c10; c[t1+1] = c11; c[t1+2] = c12;
                            c[t2] = c20; c[t2+1] = c21; c[t2+2] = c22;
                        }
            
                    for (i1 = i; i1 < i+B; ++i1)
                        for (j1 = j+p; j1 < j+B; ++j1)
                            for (k1 = k; k1 < k+B; ++k1)
                                c[i1*lc+j1] += a[i1*la+k1] * b[k1*lb+j1];
                    for (i1 = i+p; i1 < i+B; ++i1)
                        for (j1 = j; j1 < j+p; ++j1)
                            for (k1 = k; k1 < k+B; ++k1)
                                c[i1*lc+j1] += a[i1*la+k1] * b[k1*lb+j1];
                }
    // A01 x B10
    for (i = 0; i < m; i+=B)
        for (k = m1; k < n1; ++k) 
            for (j = 0; j < m2; j+=B)  
                /* B x B mini matrix multiplications */
                for (i1 = i; i1 < i+B; ++i1) {
                    register double r=a[i1 * n1 + k];
                    for (j1 = j; j1 < j+B; ++j1) 
                        c[i1 * lc + j1] += r * b[k * lb + j1];
                }
    for (i = 0; i < m; ++i) 
        for (k = 0; k < n1; ++k)
            for (j = m2; j< n2; ++j)
                c[i * lc + j] += a[i * la + k] * b[k * lb + j]; 
    for (i = m; i < n; ++i)
        for (k = 0; k < n1; ++k)
            for (j = 0; j< n2; ++j)
                c[i * lc + j] += a[i * la + k] * b[k * lb + j]; 
}

inline void
dgemm_mixed(double *a, double *b, double *c, unsigned int n, unsigned int n1, unsigned int n2) {
    dgemm_mixed(a,b,c,n,n1,n2,n1,n2,n2);
}

void
copy_matrix(double *a, double *b, unsigned int n, unsigned int m, unsigned int la, unsigned int lb) {
    for (int i = 0; i < n; ++i) 
        for (int j = 0; j < m; ++j)
            a[i*la+j] = b[i*lb+j];
}

void
minus_matrix(double *a, double *b, unsigned int n, unsigned int m, unsigned int la, unsigned int lb) {
    for (int i = 0; i < n; ++i) 
        for (int j = 0; j < m; ++j)
            a[i*la+j] -= b[i*lb+j];
}

inline void
copy_matrix(double *a, double *b, unsigned int n, unsigned int m) {
    copy_matrix(a, b, n, m, m, m);
}

bool dgemm_mixed_test(unsigned int n, unsigned int n1, unsigned int n2) {
    printf("------------------------\n");
    printf("n: %d, %d, %d\n", n, n1, n2);

    double* a = createMatrixWithRandomData(n, n1);
    double* b = createMatrixWithRandomData(n1, n2);

    double* c0 = runTest(dgemm_ikj, a, b, n, n1, n2);
    double* c = runTest(dgemm_mixed, a, b, n, n1, n2);
    if (checkEqual(c0, c, n, n2) == 0) {
        printf("%s output error\n", "dgemm_mixed");
        printMatrix(c0, n, n2);
        printMatrix(c, n, n2);
        exit(1);
    } else {
        printf("%s output checked\n", "dgemm_mixed");
    }
    free(a);  free(b);
    free(c0); free(c);
}