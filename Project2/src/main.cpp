#include <iostream>
#include "lapacke.h"
#include "cblas.h"
#include <cstring>
#include <cmath>

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

int mydgetrf(int n, double* a, int* pvt) {
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

    return 0;
}

enum dtrsm_type {
    Lower = 0, 
    Upper
};

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


#undef A

 
int main() {
    const int m = 1, n = 3;
    printf("lapack test\n");
    double a[] = 
	{
		3,-1,-1,
		4,-2,-1,
		-3,2,1
    };
    
    double aa[9];
    memcpy(aa, a, sizeof(double)*9);
    int ipiv[3], iipiv[3];
    print_matrix("a", n, n, a, n);
    
    mydgetrf(3, aa, iipiv);
    print_matrix("LU of a by myself", n, aa);


    lapack_int info = LAPACKE_dgetrf(LAPACK_ROW_MAJOR, n, n, a, n, ipiv); 
    print_matrix("LU of a", n, n, a, n); 
    // info = LAPACKE_dgetri(LAPACK_ROW_MAJOR, m, a, lda, ipiv);
    // print_matrix("inv(a)", m, n, a, lda); // inv(a)
    
    double b[] = {1, 1, 0};
    double bb[3];
    memcpy(bb, b, sizeof(double)*3);
    print_matrix("b", 1, 3, b, n);
    

    double* y = mydtrsm(Lower, 3, aa, bb, iipiv);
    double* x = mydtrsm(Upper, 3, aa, y, iipiv);
    print_matrix("my", 1, 3, y, 3);
    print_matrix("mx", 1, 3, x, 3);

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

    return 0;
}
