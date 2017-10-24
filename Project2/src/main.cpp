#include <iostream>
#include "lapacke.h"
#include "cblas.h"
#include <cstring>
#include <cmath>
#include "common.h"
#include <unistd.h>

void print_matrix(const char* desc, lapack_int m, lapack_int n, double* a, lapack_int lda ) {
    lapack_int i, j;
    printf( "\n %s\n", desc );
    for( i = 0; i < m; i++ ) {
        for( j = 0; j < n; j++ ) printf( "\t%6.2f", a[i*lda+j] );
            printf( "\n" );
    }
}

inline void print_matrix(const char* desc, int n, double* a) {
    print_matrix(desc, n, n, a, n);
}


#define A(x, y) a[(x)*n + (y)]

/*
 * This is the normal version for LU factorization
 * n - matrix size
 * a - input matrix A (output will replace that matrix)
 * pvt - used for order transform
 */
int mydgetrf(int n, double* a, int* pvt) {
    for (int j = 0; j < n; ++j) pvt[j] = j;
    for (int i = 0; i < n-1; ++i) {
        int maxind = i;
        double maxa = abs(A(i, i)); 
        for (int t = i + 1; t < n; ++t) {
            if (abs(A(t, i)) > maxa) {
                 maxa = abs(A(t, i));
                 maxind = t;
            }
        }
        if (maxa == 0.0) {
            return -1;
        } else {
            if (maxind != i) {
                int temps = pvt[i];
                pvt[i] = pvt[maxind];
                pvt[maxind] = temps;

                double tempv;
                for (int j = 0; j < n; ++j) {
                    tempv = A(i, j);
                    A(i, j) = A(maxind, j);
                    A(maxind, j) = tempv;
                }
            }
        }
    // }
    // for (int i = 0; i < n-1; ++i) {
        for (int j = i+1; j < n; ++j) {
            A(j, i) /= A(i, i);
            for (int k = i+1; k < n; ++k) 
                A(j, k) -= A(j, i) * A(i, k);
        }
    }
    return 0;
}

enum dtrsm_type {
    Lower = 0, 
    Upper
};

/* 
 * normal dtrsm for calculate Ax = B
 * t - the triangle type (Lower or Upper)
 * n - the matrix size
 * a - the input A matrix
 * b - the input B vector
 * pvt is used for order tranform
 */
double* mydtrsm(dtrsm_type t, int n, double* a, double* b, int* pvt) {
    if (t == Lower) {
        double* y = new double[n];
        y[0] = b[pvt[0]];
        for (int i = 1; i < n; ++i) {
            double sum = b[pvt[i]];
            for (int j = 0; j < i; ++j) {
                sum -= y[j] * A(i, j);
            }
            y[i] = sum;
        }
        return y;
    } else {
        double* x = new double[n];
        x[n-1] = b[n-1] / A(n-1, n-1);
        for (int i = n-2; i >= 0; --i) {
            double sum = b[i];
            for (int j = i + 1; j < n; ++j) {
                sum -= x[j] * A(i, j);
            }
            x[i] = sum / A(i, i);
        } 
        return x;
    }
}

void inv_triangle(int n, double* a) {
    for (int i = 1; i < n; ++i)
        A(i,0) = -A(i,0);

    for (int k=1; k<n-1; ++k)
        for(int i = k+1; i < n; ++i) {
            for(int j = 0; j < k; j++)
               A(i,j) -= A(k,j) * A(i,k);
            A(i,k) = -A(i,k);
        }
} 


#define LL(x, y) ll[(x)*B + (y)]

/** 
 * the highest performance version (Blocked GEPP)
 * n - matrix size
 * a - input/output matrix
 * pvt - used for order transform
 */
int hp_dgetrf(int n, double* a, int* pvt) { 
    for (int j = 0; j<n; ++j) pvt[j] = j;
    for (int i = 0; i < n-1; ++i) {
        int maxind = i;
        double maxa = fabs(A(i, i)); 
        for (int t = i + 1; t < n; ++t) {
            if (fabs(A(t, i)) > maxa) {
                 maxa = fabs(A(t, i));
                 maxind = t;
            }
        }
        if (maxa == 0.0) {
            return -1;
        } else {
            if (maxind != i) {
                int temps = pvt[i];
                pvt[i] = pvt[maxind];
                pvt[maxind] = temps;

                double tempv;
                for (int j = 0; j < n; ++j) {
                    tempv = A(i, j);
                    A(i, j) = A(maxind, j);
                    A(maxind, j) = tempv;
                }
            }
        }
    }
    const int B = 3;
    int m = n/B*B; int m1 = (n-1)/B*B; 
    double* ll = createMatrix(B, B);
    double* temp = createMatrix(n, n);
    int i;
    for (i = 0; i < m1; i += B) {
        int end = i+B;
        // apply BLAS2 version to get A(i:n, i:i+B)
        for (int t = i; t < end; ++t)
            for (int j = t+1; j < n; ++j) {
                A(j, t) /= A(t, t);
                for (int k = t+1; k < t+B; ++k) 
                    A(j, k) -= A(j, t) * A(t, k);
            }
        // get LL
        for (int p = 0; p < B; ++p)
            for (int q = 0; q < B; ++q)
                if (p == q) LL(p, q) = 1;
                else if (p < q) LL(p, q) = 0;
                else LL(p, q) = A(i+p, i+q);

        inv_triangle(B, ll); // LL^-1
        dgemm_mixed(ll, &A(i, end), temp, B, B, n-end, B, n, n-end);   // LL^-1 * A(ib:end , end+1:n)
        copy_matrix(&A(i, end), temp, B, n-end, n, n-end);  // update A(ib:end , end+1:n)
        dgemm_mixed(&A(end, i), &A(i, end), temp, n-end, B, n-end, n, n, n-end);
        minus_matrix(&A(end, end), temp, n-end, n-end, n, n-end);
    }
    for (; i < n-1; ++i) { // continue to do the unfinished part
        for (int j = i+1; j < n; ++j) {
            A(j, i) = A(j, i) / A(i, i);
            for (int k = i+1; k < n; ++k) 
                A(j, k) = A(j, k) - A(j, i) * A(i, k);
        }
    }
    free(ll); free(temp);
    return 0;
}

#undef A

// returns GFlops
inline double calculateFlops(double n, double time) {
    double all = ((n-1)*n / 2) + (n*(n-1)*(2*n-1)/3); // all float point calcuation times
    return all / time *1e-9;
}

/**
 * This is the basic speed testing
 */
int runLUTest(unsigned int n) {
    double *a, *a1, *aa, *b, *b1, *b2, *bb, n3, timed, timed2, timed3, p, p2, p3;
    int* ipiv = new int[n];
    struct timespec ts1,ts2,diff;

    /////////////////////////////////////////////////////////
    a = createMatrix(n, n); 
    a1 = createMatrix(n, n); 
    aa = createMatrixWithRandomData(n, n); 
    memcpy(a1, aa, sizeof(double)*n*n);
    b = createMatrix(1, n);
    bb = createMatrixWithRandomData(1, n);
    memcpy(b, bb, sizeof(double)*n);
    /////////////////////////////////////////////////////////

    // print_matrix("first a", n, a);

    clock_gettime(CLOCK_REALTIME, &ts1);
    lapack_int info = LAPACKE_dgetrf(LAPACK_ROW_MAJOR, n, n, a1, n, ipiv); 
    clock_gettime(CLOCK_REALTIME, &ts2);
    for(int i = 0; i < n; i++)
    {
        double tmp = b[ipiv[i]-1];
        b[ipiv[i]-1] = b[i];
        b[i] = tmp;
    }
    // print_matrix("a", n, a);
    cblas_dtrsm(CblasRowMajor, CblasLeft, CblasLower, CblasNoTrans, CblasUnit, n, 1, 1.0, a1, n, b, 1);
    cblas_dtrsm(CblasRowMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, n, 1, 1.0, a1, n, b, 1);

    //////////////////////////////////////////////////////////////////

    timespec_diff(&ts1,&ts2,&diff);
    timed = (double)(diff.tv_sec) + 1e-9 * (double)(diff.tv_nsec);
    p = calculateFlops((double)n, timed);
    printf("%d,\t%lf,\t%lf", n, p, timed);

    memcpy(a, aa, sizeof(double)*n*n);
    // print_matrix("first a1", n, a);
    clock_gettime(CLOCK_REALTIME, &ts1);
    int err_msg = mydgetrf(n, a, ipiv);
    clock_gettime(CLOCK_REALTIME, &ts2);
    double* y = mydtrsm(Lower, n, a, bb, ipiv);
    b1 = mydtrsm(Upper, n, a, y, ipiv);
    delete[] y; 
    if (!checkEqual(b, b1, n, 1)) {
        printf("lapack info: %d, our error msg: %d", info, err_msg);
        print_matrix("Right Ans:", n, a1);
        print_matrix("mydgetrf Ans:", n, a);
        
        printf("Right Ans:\n");
        printMatrix(b, 1, n); 
        printf("mydgetrf Ans:\n");
        printMatrix(b1, 1, n); 
        
        printMatrix(y, 1, n);

        printf("\nmydgetrf error when n = %d\n", n);
        delete[] ipiv;
        return -1;
    }

    //////////////////////////////////////////////////////////////////
    

    timespec_diff(&ts1,&ts2,&diff);
    timed = (double)(diff.tv_sec) + 1e-9 * (double)(diff.tv_nsec);
    p = calculateFlops((double)n, timed);
    printf("\t%lf,\t%lf", p, timed);

    memcpy(a, aa, sizeof(double)*n*n);
    clock_gettime(CLOCK_REALTIME, &ts1);
    hp_dgetrf(n, a, ipiv);
    clock_gettime(CLOCK_REALTIME, &ts2);
    y = mydtrsm(Lower, n, a, bb, ipiv);
    b2 = mydtrsm(Upper, n, a, y, ipiv);
    delete[] y; 
    if (!checkEqual(b, b2, n, 1)) {
        print_matrix("Right Ans:", n, a1);
        print_matrix("mydgetrf Ans:", n, a);
        
        printf("Right Ans:\n");
        printMatrix(b, 1, n); 
        printf("hp_dgetrf Ans:\n");
        printMatrix(b2, 1, n); 
        
        printf("\nhp_dgetrf error when n = %d\n", n);
        delete[] ipiv;
        return -1;
    }

    timespec_diff(&ts1,&ts2,&diff);
    timed = (double)(diff.tv_sec) + 1e-9 * (double)(diff.tv_nsec);
    p = calculateFlops((double)n, timed);
    printf("\t%lf,\t%lf", p, timed);

    printf("\nmydgetrf & hp_dgetrf checked when n = %d\n", n);
    delete[] ipiv;
    return 0;
}






/* It's a small test for the correctness */
void mytest() {
    dgemm_mixed_test(3, 4, 5);
    dgemm_mixed_test(30, 40, 50);
    dgemm_mixed_test(300, 600, 300);
    dgemm_mixed_test(308, 600, 38);

    double l[] = {
        1,0,0,0,
        2,1,0,0,
        3,2,1,0,
        5,4,2,1
    };
    inv_triangle(4, l);
    printMatrix(l, 4, 4);

    const int m = 1, n = 3;
    printf("lapack test\n");
    double a[] = 
	{
		3,-1,-1,
		4,-2,-1,
		-3,2,1
    };
    
    double aa[9], aaa[9];
    memcpy(aa, a, sizeof(double)*9);
    memcpy(aaa, a, sizeof(double)*9);
    int ipiv[3], iipiv[3], iiipiv[3];
    print_matrix("a", n, n, a, n);
    
    mydgetrf(3, aa, iipiv);
    print_matrix("LU of a by myself", n, aa);

    hp_dgetrf(3, aaa, iiipiv);
    print_matrix("LU of hp_dgetrf", n, aaa);

    lapack_int info = LAPACKE_dgetrf(LAPACK_ROW_MAJOR, n, n, a, n, ipiv); 
    print_matrix("LU of a", n, n, a, n); 
    // info = LAPACKE_dgetri(LAPACK_ROW_MAJOR, m, a, lda, ipiv);
    // print_matrix("inv(a)", m, n, a, lda); // inv(a)
    
    double b[] = {1, 1, 0};
    double bb[3];
    double bbb[3];
    memcpy(bb, b, sizeof(double)*3);
    memcpy(bbb, b, sizeof(double)*3);
    print_matrix("b", 1, 3, b, n);
    

    double* y = mydtrsm(Lower, 3, aa, bb, iipiv);
    double* x = mydtrsm(Upper, 3, aa, y, iipiv);
    print_matrix("my", 1, 3, y, 3);
    print_matrix("mx", 1, 3, x, 3);


    double* yy = mydtrsm(Lower, 3, aaa, bbb, iiipiv);
    double* xx = mydtrsm(Upper, 3, aaa, yy, iiipiv);
    print_matrix("myy", 1, 3, yy, 3);
    print_matrix("mxx", 1, 3, xx, 3);

    for(int i = 0; i < n; i++)
    {
        double tmp = b[ipiv[i]-1];
        b[ipiv[i]-1] = b[i];
        b[i] = tmp;
    }
    cblas_dtrsm(CblasRowMajor, CblasLeft, CblasLower, CblasNoTrans, CblasUnit, n, m, 1.0, a, n, b, m);
    print_matrix("ly", 1, 3, b, n);
    cblas_dtrsm(CblasRowMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, n, m, 1.0, a, n, b, m);
    print_matrix("lx", 1, 3, b, n);

}


static const char* help_msg = "\
-a \trun all matrix test for block size from 1000 ~ 5000\
-n N \trun one test for special matrix size N\n\
-f \tfind the best block size\n\
-t \trun my small test\n\
";

/**
* Main Function Here 
*/
int main(int argc, char **argv) {
    srand(12306);

    char ch; int n;
    while((ch = getopt(argc, argv, "an:tf")) != -1) 
        switch(ch)
        {
        case 'a': 
            for (int i = 1000; i <= 5000; i+=1000)
                runLUTest(i);
            break;
        case 'n':
            n = atoi(optarg);  
            runLUTest(n);
            break;
        case 'f':
            
            break;
        case 't':
            printf("run mytest\n"); mytest(); break;
        default:
            printf("%s", help_msg);
            return 0;
        }

    return 0;
}
