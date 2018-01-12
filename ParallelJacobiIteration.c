#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>

struct SparseMatrix
{
  int m, n;
  int *iRow;
  int *jCol;
  double *values;
};

struct Vector
{
  int n;
  double *values;
};

struct SparseMatrix* CreateMatrix(int nx, int ny) ;

void MatVec(struct SparseMatrix *A, struct Vector *x, struct Vector *y, int iStart, int iEnd) ;

int Parallel_MPI_Jacobi(
		struct SparseMatrix *A ,
		struct Vector x ,
		struct Vector r ,
		struct Vector b,
		double tol,
		int maxIter,
		int comm_sz,
		int my_rank) ;
double norm_vec(struct Vector y) ;

void Mat_vect_mult(double local_A[], double local_x[], 
      double local_y[], int local_m, int n, int local_n, 
      MPI_Comm comm);

void PrintVector(FILE *stream, struct Vector *b) ;


int main(int argc, char *argv[]) {
	if (argc <= 2) {
		printf("\n ... The code requires two input parameters nx and ny ...\n\n");
		return -1;
	}
	
	int nx = atoi(argv[1]) ;
	int ny = atoi(argv[2]) ;
	double tol = pow(10.0,-6) ;
	int comm_sz, my_rank, ii ;
	int maxIter = 500000 ;
	double local_start, local_finish, local_elapsed, elapsed ;
	
	printf(" nx %d ny %d \n", nx, ny);
	
	struct SparseMatrix *A = CreateMatrix(nx, ny) ;
	
	struct Vector x, b, r ;
	x.n = nx * ny ;
	b.n = x.n ;
	r.n = x.n ;
	x.values = (double*) malloc(sizeof(double)*x.n) ;
	b.values = (double*) malloc(sizeof(double)*b.n) ;
	r.values = (double*) malloc(sizeof(double)*r.n) ;
	for (ii = 0; ii < x.n; ++ii)
		x.values[ii] = 0.0 ;
	for (ii = 0; ii < b.n; ++ii)
		b.values[ii] = 1.0 ;
	
	MPI_Init(&argc, &argv) ;
	MPI_Comm_size(MPI_COMM_WORLD, &comm_sz) ;
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank) ;
	
	local_start = MPI_Wtime() ;
	
	int converge = Parallel_MPI_Jacobi(A, x, r, b, tol, maxIter, comm_sz, my_rank) ;
	
	local_finish = MPI_Wtime() ;
	local_elapsed = local_finish - local_start ;
	MPI_Reduce(&local_elapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD) ;
	
	MPI_Finalize() ;
	
	if (my_rank == 0) {
		printf("\nParallel time elapsed: %e\n", elapsed) ;
		
		FILE *iterations ;
		iterations = fopen("iter.txt", "w") ;
		fprintf(iterations, "%d", converge) ;
		fclose(iterations) ;
		
		FILE *time ;
		time = fopen("t_mpi.txt", "w") ;
		fprintf(time, "%e", elapsed) ;
		fclose(time) ;
	}
	
	printf("\nDid it converge? Number of iterations: %d\n", converge) ;
	
	return 0 ;
}


int Parallel_MPI_Jacobi(
		struct SparseMatrix *A ,
		struct Vector x ,
		struct Vector r ,
		struct Vector b ,
		double tol ,
		int maxIter ,
		int comm_sz ,
		int my_rank) {
	int iter, jj, iStart, iEnd ;
	double local_resNorm, sum ;
	int local_n = x.n / comm_sz ;
	iStart = local_n * my_rank ;
	iEnd = (my_rank + 1) * local_n ;
	
	
	
	for (iter = 1; iter < maxIter; ++iter) {
		MPI_Allgather(&(x.values[iStart]), iEnd-iStart, MPI_DOUBLE, (x.values), iEnd-iStart, MPI_DOUBLE, MPI_COMM_WORLD) ;
		MatVec(A,&x,&r,iStart,iEnd) ;
		
		for (jj = iStart; jj < iEnd; ++jj)
			r.values[jj] = b.values[jj] - r.values[jj] ;
		
		for (jj = iStart; jj < iEnd; ++jj)
			x.values[jj] = x.values[jj] + (r.values[jj] / 4.0) ;
		
		MPI_Allgather(&(x.values[iStart]), iEnd-iStart, MPI_DOUBLE, (x.values), iEnd-iStart, MPI_DOUBLE, MPI_COMM_WORLD) ;
		MatVec(A,&x,&r,iStart,iEnd) ;
		
		for (jj = iStart; jj < iEnd; ++jj)
			r.values[jj] = b.values[jj] - r.values[jj] ;
		
		sum = 0.0 ;
		for (jj = iStart; jj < iEnd; ++jj)
			sum += r.values[jj] * r.values[jj] ;
		local_resNorm = sum ;
		
		MPI_Allreduce(&local_resNorm, &sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD) ;
		if (sqrt(sum) <= (tol * norm_vec(b)))
			break ;
		
		
	}
	printf("\nFinal norm: %e\n", norm_vec(r)) ;
	
	if (my_rank == 0) {
		FILE *xFull ;
		xFull = fopen("xfinal.txt", "w") ;
		PrintVector(xFull, &x) ;
		fclose(xFull) ;
	}
	return iter ;
}


double norm_vec(struct Vector y) {
	int i ;
	double sum = 0.0 ;
	for (i = 0; i < y.n; ++i)
		sum += y.values[i] * y.values[i] ;
	return sqrt(sum) ;
}


void PrintVector(FILE *stream, struct Vector *b) {
	int i;
	for (i = 0; i < b->n; ++i)
		fprintf(stream, "%e\n", b->values[i]);
}


void MatVec(struct SparseMatrix *A, struct Vector *x, struct Vector *y, int iStart, int iEnd) {
	if ((A->n != x->n) || (A->m != y->n))
		return;
	int i, k;
	for (i = iStart; i < iEnd; ++i) {
		double myval = 0.0;
		for (k = A->iRow[i]; k < A->iRow[i+1]; ++k)
			myval += A->values[k] * x->values[A->jCol[k]];
		y->values[i] = myval;
	}
}


void Mat_vect_mult(
		double    local_A[]  /* in  */, 
		double    local_x[]  /* in  */, 
		double    local_y[]  /* out */,
		int       local_m    /* in  */, 
		int       n          /* in  */,
		int       local_n    /* in  */,
		MPI_Comm  comm       /* in  */) {
	double* x;
	int local_i, j;
	
	x = malloc(n*sizeof(double));
	
	MPI_Allgather(local_x, local_n, MPI_DOUBLE, x, local_n, MPI_DOUBLE, comm) ;
	
	for (local_i = 0; local_i < local_m; local_i++) {
		local_y[local_i] = 0.0;
		for (j = 0; j < n; j++)
			local_y[local_i] += local_A[local_i*n+j]*x[j];
	}
	free(x);
}  /* Mat_vect_mult */


struct SparseMatrix* CreateMatrix(int nx, int ny) {
	
	struct SparseMatrix *A = (struct SparseMatrix*) malloc(sizeof(struct SparseMatrix)*1);
	A->m = nx * ny;
	A->n = A->m;
	A->iRow = (int*) malloc(sizeof(int)*(A->m+1));
	A->iRow[0] = 0;
	
	int ix, iy;
	int count = 0;
	
	for (iy = 0; iy < ny; ++iy) {
		for (ix = 0; ix < nx; ++ix) {
			int mycount = 1;
			if (ix > 0)
				mycount += 1;
			if (iy > 0)
				mycount += 1;
			if (ix + 1 < nx)
				mycount += 1;
			if (iy + 1 < ny)
				mycount += 1;
			A->iRow[count+1] = A->iRow[count] + mycount;
			count += 1;
		}
	}
	
	int nnz = A->iRow[A->m];
	
	A->jCol = (int*) malloc(sizeof(int)*nnz);
	A->values = (double*) malloc(sizeof(double)*nnz);
	
	count = 0;
	for (iy = 0; iy < ny; ++iy) {
		for (ix = 0; ix < nx; ++ix) {
			
			int mycount = A->iRow[count];
			
			/* */
			if (iy > 0) {
				A->values[mycount] = -1.0;
				A->jCol[mycount] = count - nx;
				mycount += 1;
			}
			
			/* */
			if (ix > 0) {
				A->values[mycount] = -1.0;
				A->jCol[mycount] = count - 1;
				mycount += 1;
			}
			
			/* */
			A->values[mycount] = 4.0;
			A->jCol[mycount] = count;
			mycount += 1;
			
			/* */
			if (ix + 1 < nx) {
				A->values[mycount] = -1.0;
				A->jCol[mycount] = count + 1;
				mycount += 1;
			}
			
			/* */
			if (iy + 1 < ny) {
				A->values[mycount] = -1.0;
				A->jCol[mycount] = count + nx;
				mycount += 1;
			}
			
			/* */
			count += 1;
		}
	}
	return A;
}
