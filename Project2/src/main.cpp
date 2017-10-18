#include <iostream>
#include "lapacke.h"
#include "cblas.h"

const int m = 3, n = 3, lda = 3;

void print_matrix( char* desc, lapack_int m, lapack_int n, double* a, lapack_int lda ) {
    lapack_int i, j;
    printf( "\n %s\n", desc );
    for( i = 0; i < m; i++ ) {
            for( j = 0; j < n; j++ ) printf( " %6.2f", a[i*lda+j] );
            printf( "\n" );
    }
}

int main() {
    printf("lapack test\n");
    double a[] = 
	{
		3,-1,-1,
		4,-2,-1,
		-3,2,1
    };
    int ipiv[3];
    print_matrix("a", m, n, a, lda);
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
