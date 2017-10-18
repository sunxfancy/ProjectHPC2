#include <stdlib.h>
#include <stdio.h>
#include <time.h>

double*
createMatrix(unsigned int n) {
    return (double*) calloc(sizeof(double), n*n+1);
}

double*
createMatrixWithRandomData(unsigned int n) {
    double* data; unsigned int i;
    data = createMatrix(n);
    for (i = 0; i < n*n; ++i) {
        data[i] = (double)(rand() % 9987) / 100.0;
    }
    return data;
}

int
checkEqual(double *a, double *b, unsigned int n) {
    unsigned int i;
    for (i = 0; i< n; ++i) {
        if (a[i] != b[i]) return 0;
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

void printMatrix(double* m, unsigned int n) {
    unsigned int i, j;
    for (i = 0; i < n; ++i) {
        for (j = 0; j < n; ++j)
            printf("%.2f ", m[i*n+j]);
        printf("\n");
    }
}

typedef void (*matrixMutiply)(double *a, double *b, double *c, unsigned int n);

double* runTest(matrixMutiply func, double *a, double *b, unsigned int n) {
    double *c, timed, n3, p;
    struct timespec ts1,ts2,diff;
    c = createMatrix(n);
    clock_gettime(CLOCK_REALTIME, &ts1);
    func(a,b,c,n);
    clock_gettime(CLOCK_REALTIME, &ts2);
    timespec_diff(&ts1,&ts2,&diff);
    timed = (double)(diff.tv_sec) + 1e-9 * (double)(diff.tv_nsec);
    n3 = (double)n;
    p = 2*n3*n3*n3/timed*1e-9;
    printf("performace: %lf Gflops\n", p);
    printf("timespec is %lu s %lu ns\n", diff.tv_sec, diff.tv_nsec);
    return c;
}