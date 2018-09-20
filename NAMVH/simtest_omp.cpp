#include "mex.h"
#include "matrix.h"
#include <inttypes.h>
#include <cmath>
#include <omp.h>

//to compile, run:
//mex simtest_omp.cpp -largeArrayDims  COMPFLAGS="/openmp $COMPFLAGS"
//to compile in Linux, run:
//mex -O -g simtest_omp.cpp -largeArrayDims CXXFLAGS="\$CXXFLAGS -fopenmp" LDFLAGS="\$LDFLAGS -fopenmp"

double sumAnnz_single(double *ind_nnz, double *sim, int A_nnz,int i,int j,int N_test)
{
    int k;
    double temp_sum = 0;

    for(k=0;k<A_nnz;k++)
    {
        int temp_a = 0;
        temp_a = ind_nnz [i*A_nnz+k];
        temp_sum = temp_sum + sim[(temp_a - 1)*N_test + j];
    }
    return temp_sum;
}

void sum_column(double *ind_nnz, double*sim, int A_nnz, int i, int N_test, double *out)
{
//#pragma omp parallel for  
    for(int j=0;j<N_test;j++)
    {
        {
            out[j] = sumAnnz_single(ind_nnz, sim, A_nnz, i, j, N_test);
        }
    }
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if(nrhs!=4){
        mexErrMsgTxt("There are 4 inputs!\n");
    }
    int N_test,N_retri,num_c,A_nnz;
    double *sim,*ind_nnz,*out;
    N_test = mxGetScalar(prhs[2]);//out  ï¿½ï¿½ï¿½ï¿½
    N_retri = mxGetScalar(prhs[3]);//out  ï¿½ï¿½ï¿½ï¿½
    sim = mxGetPr(prhs[0]);//sim ï¿½ï¿½ï¿½ï¿½
    ind_nnz = mxGetPr(prhs[1]); // A ï¿½ï¿½ï¿½ï¿½
    num_c = mxGetN(prhs[0]); //ï¿½ï¿½ï¿?cluster ï¿½ï¿½ï¿½ï¿½
    A_nnz = mxGetM(prhs[1]); //ï¿½ï¿½ï¿?ind ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿?
    plhs[0] = mxCreateDoubleMatrix(N_test, N_retri, mxREAL);
    out = mxGetPr(plhs[0]);
    
    //omp_set_num_threads(8);
    
//#pragma omp parallel for  
    for(int i=0;i<N_retri;i++)
    {
        sum_column(ind_nnz, sim, A_nnz, i, N_test, &out[i*N_test]); 
    }
}

