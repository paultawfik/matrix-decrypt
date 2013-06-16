#include <stdio.h>
#include <nmmintrin.h>
#include <omp.h>

void sgemm( int m, int n, int d, float *A, float *C ) {
#pragma omp parallel 
	{
		__m128 t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, e1, e2, trans;
		register int k, j, i, limit1, limit2, limit3, limit4, x, y;
		limit1 = n / 24 * 24;
		limit2 = m / 2 * 2;
		limit3 = n / 8 * 8;
		limit4 = n / 4 * 4;
		y = m / 4 * 4;
		#pragma omp for
		for( j = 0; j < n; j++ ){ 
			for( i = 0; i < limit1; i += 24 ){
				x = j * n + i;
				t0 = _mm_loadu_ps(C + x);
				t1 = _mm_loadu_ps(C + 4 + x);
				t2 = _mm_loadu_ps(C + 8 + x);
				t3 = _mm_loadu_ps(C + 12 + x);
				t4 = _mm_loadu_ps(C + 16 + x);
				e1 = _mm_loadu_ps(C + 20 + x);
				
				for (k = 0; k < y; k += 4) {
					x = k * n + i;
					trans = _mm_load1_ps(A + j * (n + 1) + k * n);

					t5 = _mm_mul_ps(_mm_loadu_ps(A + x), trans);
					t6 = _mm_mul_ps(_mm_loadu_ps(A + 4 + x), trans);
					t7 = _mm_mul_ps(_mm_loadu_ps(A + 8 + x), trans);
					t8 = _mm_mul_ps(_mm_loadu_ps(A + 12 + x), trans);
					t9 = _mm_mul_ps(_mm_loadu_ps(A + 16 + x), trans);
					e2 = _mm_mul_ps(_mm_loadu_ps(A + 20 + x), trans);
					
					t0 = _mm_add_ps(t0, t5);
					t1 = _mm_add_ps(t1, t6);
					t2 = _mm_add_ps(t2, t7);
					t3 = _mm_add_ps(t3, t8);
					t4 = _mm_add_ps(t4, t9);
					e1 = _mm_add_ps(e1, e2);
					
					x = (k + 1) * n + i;

					trans = _mm_load1_ps(A + j * (n + 1) + (k + 1) * n);

					t5 = _mm_mul_ps(_mm_loadu_ps(A + x), trans);
					t6 = _mm_mul_ps(_mm_loadu_ps(A + 4 + x), trans);
					t7 = _mm_mul_ps(_mm_loadu_ps(A + 8 + x), trans);
					t8 = _mm_mul_ps(_mm_loadu_ps(A + 12 + x), trans);
					t9 = _mm_mul_ps(_mm_loadu_ps(A + 16 + x), trans);
					e2 = _mm_mul_ps(_mm_loadu_ps(A + 20 + x), trans);

					t0 = _mm_add_ps(t0, t5);
					t1 = _mm_add_ps(t1, t6);
					t2 = _mm_add_ps(t2, t7);
					t3 = _mm_add_ps(t3, t8);
					t4 = _mm_add_ps(t4, t9);
					e1 = _mm_add_ps(e1, e2);

					x = (k + 2) * n + i;

					trans = _mm_load1_ps(A + j * (n + 1) + (k + 2) * n);

					t5 = _mm_mul_ps(_mm_loadu_ps(A + x), trans);
					t6 = _mm_mul_ps(_mm_loadu_ps(A + 4 + x), trans);
					t7 = _mm_mul_ps(_mm_loadu_ps(A + 8 + x), trans);
					t8 = _mm_mul_ps(_mm_loadu_ps(A + 12 + x), trans);
					t9 = _mm_mul_ps(_mm_loadu_ps(A + 16 + x), trans);
					e2 = _mm_mul_ps(_mm_loadu_ps(A + 20 + x), trans);

					t0 = _mm_add_ps(t0, t5);
					t1 = _mm_add_ps(t1, t6);
					t2 = _mm_add_ps(t2, t7);
					t3 = _mm_add_ps(t3, t8);
					t4 = _mm_add_ps(t4, t9);
					e1 = _mm_add_ps(e1, e2);	
					
					x = (k + 3) * n + i;

					trans = _mm_load1_ps(A + j * (n + 1) + (k + 3) * n);

					t5 = _mm_mul_ps(_mm_loadu_ps(A + x), trans);
					t6 = _mm_mul_ps(_mm_loadu_ps(A + 4 + x), trans);
					t7 = _mm_mul_ps(_mm_loadu_ps(A + 8 + x), trans);
					t8 = _mm_mul_ps(_mm_loadu_ps(A + 12 + x), trans);
					t9 = _mm_mul_ps(_mm_loadu_ps(A + 16 + x), trans);
					e2 = _mm_mul_ps(_mm_loadu_ps(A + 20 + x), trans);

					t0 = _mm_add_ps(t0, t5);
					t1 = _mm_add_ps(t1, t6);
					t2 = _mm_add_ps(t2, t7);
					t3 = _mm_add_ps(t3, t8);
					t4 = _mm_add_ps(t4, t9);
					e1 = _mm_add_ps(e1, e2);
				}

				for (k; k < limit2; k += 2) {
					x = k * n + i;
					trans = _mm_load1_ps(A + j * (n + 1) + k * n);

					t5 = _mm_mul_ps(_mm_loadu_ps(A + x), trans);
					t6 = _mm_mul_ps(_mm_loadu_ps(A + 4 + x), trans);
					t7 = _mm_mul_ps(_mm_loadu_ps(A + 8 + x), trans);
					t8 = _mm_mul_ps(_mm_loadu_ps(A + 12 + x), trans);
					t9 = _mm_mul_ps(_mm_loadu_ps(A + 16 + x), trans);
					e2 = _mm_mul_ps(_mm_loadu_ps(A + 20 + x), trans);

					
					t0 = _mm_add_ps(t0, t5);
					t1 = _mm_add_ps(t1, t6);
					t2 = _mm_add_ps(t2, t7);
					t3 = _mm_add_ps(t3, t8);
					t4 = _mm_add_ps(t4, t9);
					e1 = _mm_add_ps(e1, e2);
					
					x = (k + 1) * n + i;

					trans = _mm_load1_ps(A + j * (n + 1) + (k + 1) * n);

					t5 = _mm_mul_ps(_mm_loadu_ps(A + x), trans);
					t6 = _mm_mul_ps(_mm_loadu_ps(A + 4 + x), trans);
					t7 = _mm_mul_ps(_mm_loadu_ps(A + 8 + x), trans);
					t8 = _mm_mul_ps(_mm_loadu_ps(A + 12 + x), trans);
					t9 = _mm_mul_ps(_mm_loadu_ps(A + 16 + x), trans);
					e2 = _mm_mul_ps(_mm_loadu_ps(A + 20 + x), trans);

					t0 = _mm_add_ps(t0, t5);
					t1 = _mm_add_ps(t1, t6);
					t2 = _mm_add_ps(t2, t7);
					t3 = _mm_add_ps(t3, t8);
					t4 = _mm_add_ps(t4, t9);
					e1 = _mm_add_ps(e1, e2);
				}

				for (k; k < m; k++) {
					
					x = k * n + i;
					trans = _mm_load1_ps(A + j * (n + 1) + k * n);

					t5 = _mm_mul_ps(_mm_loadu_ps(A + x), trans);
					t6 = _mm_mul_ps(_mm_loadu_ps(A + 4 + x), trans);
					t7 = _mm_mul_ps(_mm_loadu_ps(A + 8 + x), trans);
					t8 = _mm_mul_ps(_mm_loadu_ps(A + 12 + x), trans);
					t9 = _mm_mul_ps(_mm_loadu_ps(A + 16 + x), trans);
					e2 = _mm_mul_ps(_mm_loadu_ps(A + 20 + x), trans);

					t0 = _mm_add_ps(t0, t5);
					t1 = _mm_add_ps(t1, t6);
					t2 = _mm_add_ps(t2, t7);
					t3 = _mm_add_ps(t3, t8);
					t4 = _mm_add_ps(t4, t9);
					e1 = _mm_add_ps(e1, e2);
				}
				
				x = j * n + i;
				_mm_storeu_ps(C + x, t0);
				_mm_storeu_ps(C + 4 + x, t1);
				_mm_storeu_ps(C + 8 + x, t2);
				_mm_storeu_ps(C + 12 + x, t3);
				_mm_storeu_ps(C + 16 + x, t4);
				_mm_storeu_ps(C + 20 + x, e1);
			}
			
			for(i; i < limit3; i += 8 ) {
				t0 = _mm_loadu_ps(C + i + j * n);
				t1 = _mm_loadu_ps(C + i + 4 + j * n);
				
				for (k = 0; k < limit2; k += 2) {
					trans = _mm_load1_ps(A + j * (n + 1) + k * n);

					t5 = _mm_mul_ps(_mm_loadu_ps(A + 0 + i + k * n), trans);
					t6 = _mm_mul_ps(_mm_loadu_ps(A + 4 + i + k * n), trans);
					
					t0 = _mm_add_ps(t0, t5);
					t1 = _mm_add_ps(t1, t6);

					trans = _mm_load1_ps(A + j * (n + 1) + (k + 1) * n);

					t7 = _mm_mul_ps(_mm_loadu_ps(A + 0 + i + (k + 1) * n), trans);
					t8 = _mm_mul_ps(_mm_loadu_ps(A + 4 + i + (k + 1) * n), trans);

					t0 = _mm_add_ps(t0, t7);
					t1 = _mm_add_ps(t1, t8);
				}
				
				if (k < m) {
					trans = _mm_load1_ps(A + j * (n + 1) + k * n);

					t5 = _mm_mul_ps(_mm_loadu_ps(A + 0 + i + k * n), trans);
					t6 = _mm_mul_ps(_mm_loadu_ps(A + 4 + i + k * n), trans);

					t0 = _mm_add_ps(t0, t5);
					t1 = _mm_add_ps(t1, t6);

				}
				
				_mm_storeu_ps(C + 0 + i + j * n, t0);
				_mm_storeu_ps(C + 4 + i + j * n, t1);
			}

			for(i; i < limit4; i += 4 ) {
				t0 = _mm_loadu_ps(C + i + j * n);
				
				for (k = 0; k < limit2; k += 2) {
					trans = _mm_load1_ps(A + j * (n + 1) + k * n);
					e1 = _mm_load1_ps(A + j * (n + 1) + (k + 1) * n);

					t5 = _mm_mul_ps(_mm_loadu_ps(A + 0 + i + k * n), trans);
					t7 = _mm_mul_ps(_mm_loadu_ps(A + 0 + i + (k + 1) * n), e1);
					t0 = _mm_add_ps(t0, t5);
					t0 = _mm_add_ps(t0, t7);

				}
				
				if (k < m) {
					trans = _mm_load1_ps(A + j * (n + 1) + k * n);

					t5 = _mm_mul_ps(_mm_loadu_ps(A + 0 + i + k * n), trans);
					t0 = _mm_add_ps(t0, t5);
				}
				
				_mm_storeu_ps(C + 0 + i + j * n, t0);
			}

			for(i; i < n; i++) {
				for( k = 0; k < m; k++ ) {
					C[i + j * n] += A[i + k * n] * A[j * (n + 1)+k * n];
				}
			}
		}
	}
}
