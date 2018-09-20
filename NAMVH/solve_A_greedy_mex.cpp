#include "mex.h"
#include "matrix.h"
#include <omp.h>
#include <inttypes.h>
#include<vector>

using namespace std;

//to compile, run:
//mex solve_A_greedy_mex.cpp -largeArrayDims  COMPFLAGS="/openmp $COMPFLAGS"
//to compile in Linux, run:
//mex -O -g solve_A_greedy_mex.cpp -largeArrayDims CXXFLAGS="\$CXXFLAGS -fopenmp" LDFLAGS="\$LDFLAGS -fopenmp"

void sum_add(double*U, double*diagU, double*f, int*sel, vector<int>&not_sel, int n, vector<double>&sum) {

int row;
int M = sum.size();
for(int i=0;i<not_sel.size();i++){
    row = not_sel[i];
    //mexPrintf("%d \n",row+1);
    sum[row] = f[row]+diagU[row];
    if(n>0){
        for(int j=0;j<n;j++){
            sum[row] += 2*U[sel[j]*M+row];
            //mexPrintf("row=%d,col=%d,val=%f\n",row+1,sel[j]+1,U[sel[j]*M+row]);
        }
    }
}
/*for(int i=0;i<sum.size();i++){
    mexPrintf("%f ",sum[i]);
}
mexPrintf("\n");*/

}


void greedy_a(double*U, double*diagU, double*f, int M, int A_nnz, int*sel) {

int row;
vector<int> not_sel(M);
vector<double> sum(M,0);
double max_val;
for(int i=0;i<M;i++)
{
    not_sel[i] = i;
    //mexPrintf("%d\n",not_sel[i]);
}

/*for(int i=0;i<M;i++){
    for(int j=0;j<M;j++)
        mexPrintf("%f ",U[j*M+i]);
    mexPrintf("\n");
}
mexPrintf("\n");*/


for(int i=0;i<A_nnz;i++){
    sum_add(U,diagU,f,sel,not_sel,i,sum);
    max_val = -DBL_MAX;
    int del;
    for(int k=0;k<M-i;k++)
    {
        row = not_sel[k];
        if(sum[row]>max_val){
            max_val = sum[row];
            sel[i] = row;
            del = k;
        }
    }
    //mexPrintf("del=%d,size=%d\n",del+1,not_sel.size());
    not_sel.erase(not_sel.begin()+del);
}
//mexPrintf("%d \n", sel[0]);
}

void mexFunction(int nlhs, mxArray *plhs[],int nrhs, const mxArray *prhs[])
{
double*U;
double*F;
int*sel;
int M, N, A_nnz;

if(nrhs!=3){
    mexErrMsgTxt("There are 3 inputs!\n");
    mexPrintf("usage:\tsel=solve_A_greedy_mex(U,F,A_nnz), A_nnz is the number of nnz entries\n");
}

if(!mxIsDouble(prhs[0])) mexErrMsgTxt("Matrix U must be double!\n");
if(!mxIsDouble(prhs[1])) mexErrMsgTxt("Vector f must be double!\n");
if(mxGetM(prhs[0])!=mxGetN(prhs[0])) mexErrMsgTxt("U must be square matrix!\n");
if(mxGetM(prhs[0])!=mxGetM(prhs[1])) mexErrMsgTxt("size of U and F do not match!\n");
if(mxGetM(prhs[2])!=1||mxGetN(prhs[2])!=1) mexErrMsgTxt("A_nnz must be a scalar!\n");

M = mxGetM(prhs[0]);//dictionary size
N = mxGetN(prhs[1]);//number of samples
A_nnz = (int)mxGetScalar(prhs[2]);//number of nnz entries
//mexPrintf("M=%d, N=%d, A_nnz=%d\n",M,N,A_nnz);

U = (double*)mxGetPr(prhs[0]);
F = (double*)mxGetPr(prhs[1]);
plhs[0] = mxCreateNumericMatrix(A_nnz,N,mxINT32_CLASS,mxREAL);
sel = (int*)mxGetPr(plhs[0]);

double*diagU = new double[M];
for(int i=0;i<M;i++){
    diagU[i] = U[i*M+i];
    //mexPrintf("%f\n",diagU[i]);
}

//use OpenMP
#pragma omp parallel for
for(int j=0;j<N;j++){
    greedy_a(U, diagU, &F[j*M], M, A_nnz, &sel[j*A_nnz]);
}

int num = A_nnz*N;
#pragma omp parallel for
for(int i=0;i<num;i++)
    sel[i]++;

delete[] diagU;

}