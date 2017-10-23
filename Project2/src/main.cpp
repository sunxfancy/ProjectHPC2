#include <iostream>
#include "lapacke.h"
#include "cblas.h"
#include <cstring>
#include <cmath>
#include "common.h"

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

int hp_dgetrf(int n, double* a, int* pvt) { 
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
    }
    const int B = 30;
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
                for (int k = t+1; k < t+B; ++k) {
                    A(j, k) -= A(j, t) * A(t, k);
                }
            }
        // printf("-------------------\n");
        // printMatrix(&A(i, i), 3, 3, n);
        
        // get LL
        for (int p = 0; p < B; ++p)
            for (int q = 0; q < B; ++q)
                if (p == q) LL(p, q) = 1;
                else if (p < q) LL(p, q) = 0;
                else LL(p, q) = A(i+p, i+q);
        // printf("-------------------\n");
        // printMatrix(ll, B, B, B);
        inv_triangle(B, ll); // LL^-1
        // printf("-------------------\n");
        // printMatrix(ll, B, B, B);

        dgemm_mixed(ll, &A(i, end), temp, B, B, n-end, B, n, n-end);   // LL-1 * A(ib:end , end+1:n)
        // printf("-------------------\n");
        // printMatrix(temp, B, n-end, n-end);
        copy_matrix(&A(i, end), temp, B, n-end, n, n-end);  // update A(ib:end , end+1:n)
        
        // printf("-------------------\n");
        // printMatrix(&A(i, i), 3, 3, n);

        dgemm_mixed(&A(end, i), &A(i, end), temp, n-end, B, n-end, n, n, n-end);
        minus_matrix(&A(end, end), temp, n-end, n-end, n, n-end);
    }
    for (; i < n-1; ++i) { // continue to do the unfinished part
        for (int j = i+1; j < n; ++j) {
            A(j, i) /= A(i, i);
            for (int k = i+1; k < n; ++k) {
                A(j, k) -= A(j, i) * A(i, k);
            }
        }
    }
    free(ll); free(temp);
    return 0;
}



#undef A

 
int main() {
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

    return 0;
}
