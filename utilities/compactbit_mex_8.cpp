#include "mex.h"
#include "matrix.h"
#include <omp.h>
#include <inttypes.h>
#include <cmath>

//to compile, run:
//mex compactbit_mex_8.cpp -largeArrayDims  COMPFLAGS="/openmp $COMPFLAGS"
//to compile in Linux, run:
//mex -O -g compactbit_mex_8.cpp -largeArrayDims CXXFLAGS="\$CXXFLAGS -fopenmp" LDFLAGS="\$LDFLAGS -fopenmp"

template <typename T>
void set_one_column(T*B, mxLogical*H, uint32_t*ks, uint32_t wordsize, int nbits) {
    #pragma omp parallel for
    for(int i=0;i<nbits;i++)
        if(H[i]) B[ks[i]] |= ((T)1) << ((T)i)%wordsize;
}

void mexFunction(int nlhs, mxArray *plhs[],int nrhs, const mxArray *prhs[])
{
uint8_t* B;
mxLogical *H;//H is a logical matrix
uint32_t nbits, nwords, N, *ks;
double wordsize;
int i, j;

if(nrhs!=2){
    mexErrMsgTxt("There are 2 inputs!\n");
    mexPrintf("usage:\tB=compactbit_mex_32(H), where H is a binary nbits*N matrix\n");
}

if(!mxIsLogical(prhs[0])) mexErrMsgTxt("Matrix H must be logical");
if(mxGetM(prhs[1])!=mxGetN(prhs[1])) mexErrMsgTxt("wordsize must be scalar!\n");

H = mxGetLogicals(prhs[0]);
nbits = mxGetM(prhs[0]);
N = mxGetN(prhs[0]);
if(N>pow(2.0,31.0)) mexErrMsgTxt("columns of H must be less than 2^31 for using OpenMP!\n");
wordsize = mxGetScalar(prhs[1]);
if(wordsize!=floor(wordsize)) mexErrMsgTxt("wordsize must be an integer!\n");
if(!(wordsize==8|wordsize==16|wordsize==32|wordsize==64))
    mexErrMsgTxt("wordsize must be 8 or 16 or 32 or 64\n");

nwords = ceil(nbits/wordsize);
plhs[0] = mxCreateNumericMatrix(nwords,N,mxUINT8_CLASS,mxREAL);
B = (uint8_t*)mxGetPr(plhs[0]);

ks = (uint32_t*)mxMalloc(sizeof(mxUINT32_CLASS)*nbits);
//mexPrintf("size=%f\n",1.0*sizeof(mxUINT32_CLASS));
for(i=0;i<nbits;i++){
    ks[i] = ceil((i+1)/wordsize)-1;
    //mexPrintf("%d\n",ks[i]);
}
//mexPrintf("nbits=%d, N=%d, wordsize=%f, nwords=%d\n",nbits,N,wordsize,nwords);

//use OpenMP
#pragma omp parallel for
for(j=0;j<N;j++){
    set_one_column(&B[j*nwords], &H[j*nbits], ks, wordsize, nbits);
    /*for(i=0;i<nwords;i++){
        printf("i=%d,j=%d,bij=%f\n",i,j,1.0*j*nwords+i);
    }*/
}

mxFree(ks);

}