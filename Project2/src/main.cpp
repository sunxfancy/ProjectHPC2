#include <iostream>
#include "lapacke.h"
#include "cblas.h"



void print_matrix(const char* desc, lapack_int m, lapack_int n, double* a, lapack_int lda ) {
    lapack_int i, j;
    printf( "\n %s\n", desc );
    for( i = 0; i < m; i++ ) {
            for( j = 0; j < n; j++ ) printf( " %6.2f", a[i*lda+j] );
            printf( "\n" );
    }
}

void print_matrix(const char* desc, int n, double* a) {
    print_matrix(desc, n, n, a, n);
}

#define A(x, y) a[(x)*n + (y)]

int mydgetrf(int n, double* a, int* P) {
    int* pvt;

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
            pvt = new int[n];
            for (int j = 0; j<n; ++j) pvt[j] = j;
            if (maxind != i) {
                int temps = pvt[i];
                pvt[i] = pvt[maxind];
                pvt[maxind] = temps;

                for (int j = 0; j < n; ++j) {
                    temps = A(i, j);
                    A(i, j) = A(maxind, j);
                    A(maxind, j) = temps;
                }
            }
        }
        for (int j = i+1; j < n; ++j) {
            A(j, i) /= A(i, i);
            for (int k = i+1; k < n; ++k) {
                A(j, k) -= A(j, i) * A(i, k);
            }
        }
    }

    P = pvt;
    return 0;
}

#undef A

 
int main() {
    const int m = 3, n = 3, lda = 3;
    printf("lapack test\n");
    double a[] = 
	{
		3,-1,-1,
		4,-2,-1,
		-3,2,1
    };
    
    double aa[9];
    memcpy(aa, a, sizeof(double)*9);
    int ipiv[3];
    print_matrix("a", m, n, a, lda);
    
    mydgetrf(3, aa, ipiv);
    print_matrix("LU of a by myself", 3, aa);


    lapack_int info = LAPACKE_dgetrf(LAPACK_ROW_MAJOR, m, n, a, lda, ipiv); // 由于L的对角线都是1，则这里对角线列出的是U的值
    print_matrix("LU of a", m, n, a, lda); 
    // info = LAPACKE_dgetri(LAPACK_ROW_MAJOR, m, a, lda, ipiv);
    // print_matrix("inv(a)", m, n, a, lda); // inv(a)
    
    double b[] = {1, 1, 0};
    cblas_dtrsm(CblasRowMajor, CblasRight, CblasUpper, CblasNoTrans, CblasNonUnit, m, n, 1, a, lda, b, 3);
    print_matrix("b", 1, 3, b, 3);

    for (int i =0; i<3; ++i) a[i*3+i] = 1;

    double c[] = {1, 2, 4};
    cblas_dtrsm(CblasRowMajor, CblasRight, CblasLower, CblasNoTrans, CblasUnit, m, n, 1, a, lda, c, 3);
    print_matrix("c", 1, 3, c, 3);

    return 0;
}
