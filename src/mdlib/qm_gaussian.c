/*
 * 
 *                This source code is part of
 * 
 *                 G   R   O   M   A   C   S
 * 
 *          GROningen MAchine for Chemical Simulations
 * 
 *                        VERSION 3.2.0
 * Written by David van der Spoel, Erik Lindahl, Berk Hess, and others.
 * Copyright (c) 1991-2000, University of Groningen, The Netherlands.
 * Copyright (c) 2001-2004, The GROMACS development team,
 * check out http://www.gromacs.org for more information.
 
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 * 
 * If you want to redistribute modifications, please consider that
 * scientific software is very special. Version control is crucial -
 * bugs must be traceable. We will be happy to consider code for
 * inclusion in the official distribution, but derived work must not
 * be called official GROMACS. Details are found in the README & COPYING
 * files - if they are missing, get the official version at www.gromacs.org.
 * 
 * To help us fund GROMACS development, we humbly ask that you cite
 * the papers on the package - you can find them in the top README file.
 * 
 * For more info, check our website at http://www.gromacs.org
 * 
 * And Hey:
 * GROwing Monsters And Cloning Shrimps
 */
#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#ifdef GMX_QMMM_GAUSSIAN

#include <math.h>
#include "sysstuff.h"
#include "typedefs.h"
#include "macros.h"
#include "smalloc.h"
#include "assert.h"
#include "physics.h"
#include "macros.h"
#include "vec.h"
#include "force.h"
#include "invblock.h"
#include "confio.h"
#include "names.h"
#include "network.h"
#include "pbc.h"
#include "ns.h"
#include "nrnb.h"
#include "bondf.h"
#include "mshift.h"
#include "txtdump.h"
#include "copyrite.h"
#include "qmmm.h"
#include <stdio.h>
#include <string.h>
#include "gmx_fatal.h"
#include "typedefs.h"
#include "../tools/eigensolver.h"
#include <stdlib.h>

#include "do_fit.h"
/* eigensolver stuff */
#include "sparsematrix.h"

#ifndef F77_FUNC
#define F77_FUNC(name,NAME) name ## _
#endif

//#include "gmx_lapack.h"
//#include "gmx_arpack.h"


#include <complex.h>
#ifdef I
#undef I
#endif
#define IMAG _Complex_I

#ifdef MKL
#include <mkl_types.h>
#include <mkl_lapack.h>
#else
#include <lapacke.h>
#endif

#define AU2PS (2.418884326505e-5) /* atomic unit of time in ps */
#define VEL2AU (AU2PS/BOHR2NM)
#define MP2AU (1837.36)           /* proton mass in atomic units */

#include <time.h>

/* Uncomment to print NACs*/
//#define nacs_flag true

typedef double complex dplx;

static double dottrrr(int n, real *f, rvec *A, rvec *B);

/* scalar gromacs vector product */
static void multsv(int n, dvec *R, real f, dvec *A);

/* matrix vector product */
static void multmv(int n, dplx *R, dplx *M, dplx *V);

/* multiply vector by exponential of diagonal matrix given by 
 *  * its diagonal as f*w */
static void multexpdiag(int n, dplx *r, dplx f, double *w, dplx *v);

/* hermitian adjoint of matrix */
static void dagger(int n, dplx *Mt, dplx *M);

/* diagonalize hermitian matrix using lapack/MKL routine zheev */
static void diag(int n, double *w, dplx *V, dplx *M);

/* integrate wavefunction for one MD timestep */
//static void propagate(t_QMrec *qm, t_MMrec *mm, double *QMener);

static double calc_coupling(int J, int K, double dt, int dim, double *vec, double *vecold);

/* used for the fssh algo */

/* \sum_i A_i B_i */
static real dot(int n, real *A, real *B){
  int 
    i;
  real 
    res = 0.0;
  for (i=0; i<n; i++){
    res += A[i]*B[i];
  }
  return res;
}
/* dot product of gromacs vectors with addional factor
 *  * \sum_i f_i rvec A[i] * rvec B[i]
 *   */

static dplx dot_complex(int n, dplx *A, dplx *B){
  int 
    i;
  dplx 
    res = 0.0+IMAG*0.0;
  for (i=0; i<n; i++){
//    res += A[i]*B[i];
    res += conj(A[i])*B[i];
  }
  return res;
}


static double dottrrr(int n, real *f, rvec *A, rvec *B){
  int 
    i;
  double 
    res = 0.0;
  for (i=0; i<n; ++i){
    res += f[i]*A[i][XX]*B[i][XX];
    res += f[i]*A[i][YY]*B[i][YY];
    res += f[i]*A[i][ZZ]*B[i][ZZ];
  }
  return res;
}

/* scalar gromacs vector product */
static void multsv(int n, dvec *R, real f, dvec *A){
  int 
    i;
  for (i=0; i<n; i++){
    R[i][XX] = f*A[i][XX];
    R[i][YY] = f*A[i][YY];
    R[i][ZZ] = f*A[i][ZZ];
  }
}

/* matrix vector product R = M*V, double complex */
static void multmv(int n, dplx *R, dplx *M, dplx *V){
  int 
    i, j;
  for (i=0; i<n; i++)
  {
    R[i] = 0.0;
    for (j=0; j<n; j++)
    {
      R[i] += M[i + j*n]*V[j];
    }
  }
}


static void multexpdiag(int n, dplx *r, dplx f, double *w, dplx *v){
  int 
    i;
  for (i=0; i<n; i++){
    r[i] = cexp(f*w[i])*v[i];
  } 
}

/* hermitian adjoint of matrix */
static void dagger(int n, dplx *Mt, dplx *M){
  int 
    i, j;
  for (i=0; i<n; i++){
    for (j=0; j<n; j++){
      Mt[i + j*n] = conj(M[j + i*n]);
    }
  }
}

/* diagonalize hermitian matrix using lapack/MKL routine zgeev */
static void diag_complex(int n, dplx *w, dplx *V, dplx *M){
  char
    jobz, uplo,jobvl,jobvr;
  int
    i, j, lwork, info;
  double
    *rwork;
#ifdef MKL
  /* If compilation with MKL linear algebra libraries  */
    MKL_Complex16
      *M_mkl,*w_mkl, *work,*vr,*vl;
#else
  /* else compilation with lapack linear algebra libraries  */
    lapack_complex_double
      *M_mkl,*w_mkl, *work,*vr,*vl;
#endif

  lwork = 2*n;
  jobvl = 'N';
  jobvr = 'V';
  snew(vr,n*n);
  snew(vl,n*n);

  snew(M_mkl, n*n);
  snew(w_mkl,n);
  snew(work, lwork);
  snew(rwork, 3*n);

  for (i=0; i<n; i++){
#ifdef MKL 
    w_mkl[i].real = creal(w[i]);
    w_mkl[i].imag = cimag(w[i]);
#else 
    w_mkl[i] = w[i];
#endif
    for (j=0; j<n; j++){ 
#ifdef MKL 
      M_mkl[j + i*n].real = creal(M[j + i*n]);
      M_mkl[j + i*n].imag = cimag(M[j + i*n]);
#else 
      M_mkl[j + i*n] = M[j + i*n];
#endif
    }
  }
#ifdef MKL 
  zgeev(&jobvl,&jobvr,&n, M_mkl, &n, w_mkl, vl,&n,vr,&n,work, &lwork, rwork, &info);
//  zheev(&jobz, &uplo, &n, M_mkl, &n, w, work, &lwork, rwork, &info);
#else 
  F77_FUNC(zgeev,ZGEEV)(&jobvl,&jobvr,&n, M_mkl, &n, w_mkl, vl,&n,vr,&n,work, &lwork, rwork, &info);
  if (info != 0){
    gmx_fatal(FARGS, "Lapack returned error code: %d in zheev", info);
  }
#endif   

//  fprintf(stderr,"\n\n From diag_complex, vr:\n\n");
//  printM_complex(n,vr);

//  fprintf(stderr,"\n\n From diag_complex, vl:\n\n");
//  printM_complex(n,vl);

  for (i=0; i<n; i++){
#ifdef MKL 
    w[i] = w_mkl[i].real+IMAG*w_mkl[i].imag;
#else 
    w[i] = creal(w_mkl[i])+IMAG*cimag(w_mkl[i]);
#endif 
    for (j=0; j<n; j++){
#ifdef MKL 
      V[j + i*n] = vr[j + i*n].real + IMAG*vr[j + i*n].imag;
#else 
      V[j + i*n] = creal(vr[j + i*n]) + IMAG*cimag(vr[j + i*n]);
#endif 
    }
  }
  sfree(vl);
  sfree(vr);
  sfree(rwork);
  sfree(M_mkl);
  sfree(w_mkl);
  sfree(work);
}

/* diagonalize hermitian matrix using MKL routine zheev */
static void diag(int n, double *w, dplx *V, dplx *M)
{
  char
    jobz, uplo;
  int
    i, j, lwork, info;
  double
    *rwork;
#ifdef MKL
  MKL_Complex16
    *M_mkl, *work;
#else
  lapack_complex_double
    *M_mkl, *work;
#endif

  jobz = 'V';
  uplo = 'U';
  lwork = 2*n;

  snew(M_mkl, n*n);
  snew(work, lwork);
  snew(rwork, 3*n);

  for (i=0; i<n; i++){
    for (j=0; j<n; j++){
#ifdef MKL
      M_mkl[j + i*n].real = creal(M[j + i*n]);
      M_mkl[j + i*n].imag = cimag(M[j + i*n]);
#else
      M_mkl[j + i*n] = M[j + i*n];
#endif
    }
  }
#ifdef MKL
  zheev(&jobz, &uplo, &n, M_mkl, &n, w, work, &lwork, rwork, &info);
#else
  F77_FUNC(zheev,ZHEEV)(&jobz, &uplo, &n, M_mkl, &n, w, work, &lwork, rwork, &info);
  if (info != 0){
    gmx_fatal(FARGS, "Lapack returned error code: %d in zheev", info);
  }
#endif
  for (i=0; i<n; i++){
    for (j=0; j<n; j++){
#ifdef MKL
      V[j + i*n] = M_mkl[j + i*n].real + IMAG*M_mkl[j + i*n].imag;
#else
      V[j + i*n] = creal(M_mkl[j + i*n]) + IMAG*cimag(M_mkl[j + i*n]);
#endif
    }
  }

  sfree(rwork);
  sfree(M_mkl);
  sfree(work);
}

static double calc_coupling(int J, int K, double dt, int dim, double *vec, double *vecold){
  double 
    coupling=0;
  coupling = 1./(2.*dt) * (dot(dim,&vecold[J*dim],&vec[K*dim]) - dot(dim,&vec[J*dim],&vecold[K*dim]));
//  fprintf(stderr,"coupling between %d and %d: %lf\n", J,K,coupling);
  return (coupling);
//  return 1./(2.*dt) * (dot(dim,&vecold[J*dim],&vec[K*dim]) - dot(dim,&vec[J*dim],&vecold[K*dim]));
}

static dplx calc_coupling_complex(int J, int K, double dt, int dim, dplx *vec, dplx *vecold){
  double 
    coupling=0;
  coupling = 1./(2.*dt) * (dot_complex(dim,&vecold[J*dim],&vec[K*dim]) - dot_complex(dim,&vec[J*dim],&vecold[K*dim]));
  return (coupling);
}

static void transposeM (int ndim, double *A, double *At){
  int
    i,j;
  for (i=0;i<ndim;i++){
    for (j=0;j<ndim;j++){
      At[i*ndim+j]=A[j*ndim+i];
    }
  }
}

static void transposeM_complex (int ndim, dplx *A, dplx *At){
/* need to change the sign of the imaginary componennt!!! */
  int
    i,j;
  for (i=0;i<ndim;i++){
    for (j=0;j<ndim;j++){
      At[i*ndim+j]=conj(A[j*ndim+i]);
    }
  }
}

static void MtimesV(int ndim, double *A, double *b, double *c){
 /* c = A.b */
  int
    i,k;

  for(i=0;i<ndim;i++){
    c[i]=0;
    for(k=0;k<ndim;k++){
      c[i]+=A[i*ndim+k]*b[k];
    }
  }
}

static void MtimesV_complex(int ndim, dplx *A, dplx *b, dplx *c){
 /* c = A.b */
  int
    i,k;

  for (i = 0 ; i < ndim ; i++ ){
    c[i] = 0;
    for ( k = 0 ; k < ndim ; k++ ){
      c[i] += A[i*ndim+k]*b[k];
    }
  }
}

static void MtimesM (int ndim,double *A,double *B, double *C){
  int
    i,j,k;
     
  for ( i = 0 ; i< ndim ; i++ ){
    for ( j = 0 ; j < ndim ; j++){
      C[i*ndim+j]=0.0;
      for (k=0;k<ndim;k++){
        C[i*ndim+j]+=A[i*ndim+k]*B[k*ndim+j];
      }
    }
  }
}

static void M_complextimesM_complex (int ndim, dplx *A,dplx *B,dplx *C){
  int
    i,j,k;

  for ( i = 0 ; i< ndim ; i++ ){
    for ( j = 0 ; j < ndim ; j++){
      C[i*ndim+j]=0.0;
      for (k=0;k<ndim;k++){
        C[i*ndim+j]+=A[i*ndim+k]*B[k*ndim+j];
      }
    }
  }
}

static void MtimesM_complex (int ndim, double *A,dplx *B,dplx *C){
  int
    i,j,k;

  for ( i = 0 ; i< ndim ; i++ ){
    for ( j = 0 ; j < ndim ; j++){
      C[i*ndim+j]=0.0;
      for (k=0;k<ndim;k++){
        C[i*ndim+j]+=A[i*ndim+k]*B[k*ndim+j];
      }
    }
  }
}

static void invsqrtM (int ndim, double *A, double *B){
/* B is A^-1/2 */
  int
    i;
  dplx
    *wmatrix,*V,*Vt,*temp,*newA;
  double
    *w;
  snew(newA,ndim*ndim);
  snew(w,ndim);
  snew(wmatrix,ndim*ndim);
  snew(V,ndim*ndim);
  snew(Vt,ndim*ndim);
  snew(temp,ndim*ndim);
  /* diagonalize */

  for(i=0;i<ndim*ndim;i++){
    newA[i]=A[i];
  };

  diag(ndim,w,V,newA);
//  eigensolver(A,ndim,0,ndim,w,V);

  /* take the inverse square root of the diagonalo elements */
  for ( i = 0 ; i < ndim ; i++){
    wmatrix[i*ndim+i] = 1.0/csqrt(w[i]);
  }

  dagger(ndim, Vt, V);

  /* multiple from the left by Vt */
  M_complextimesM_complex(ndim,Vt,wmatrix,temp);

  /* and from the right by V */
  M_complextimesM_complex(ndim,temp,V,wmatrix);
  for (i = 0 ; i< ndim*ndim ; i++ ){
    B[i] = creal(wmatrix[i]);
  }
  free(Vt);
  free(V);
  free(w);
  free(wmatrix);
  free(temp);
  free(newA);
}

static void invsqrtM_complex(int ndim, dplx *A, dplx *B){
/* B is A^-1/2 */
  int
    i;
  dplx
    *wmatrix,*V,*Vt,*temp,*newA;
  double
    *w;
  snew(newA,ndim*ndim);
  snew(w,ndim);
  snew(wmatrix,ndim*ndim);
  snew(V,ndim*ndim);
  snew(Vt,ndim*ndim);
  snew(temp,ndim*ndim);

  /* diagonalize */
  for(i=0;i<ndim*ndim;i++){
    newA[i]=A[i];
  };
  diag(ndim,w,V,newA);

  /* take the inverse square root of the diagonalo elements */
  for ( i = 0 ; i < ndim ; i++){
    wmatrix[i*ndim+i] = 1.0/csqrt(w[i]);
  }
  dagger(ndim, Vt, V);

  /* multiple from the left by Vt */
  M_complextimesM_complex(ndim,Vt,wmatrix,temp);

  /* and from the right by V */
  M_complextimesM_complex(ndim,temp,V,wmatrix);
  for (i = 0 ; i< ndim*ndim ; i++ ){
    B[i] = wmatrix[i];
  }
  free(Vt);
  free(V);
  free(w);
  free(wmatrix);
  free(temp);
  free(newA);
}

static void expM_complex(int ndim, dplx *A, dplx *expA){
  /* expA = exp(A)*/
  int
    i,j;
  dplx
    *wmatrix,*V,*Vt,*temp;
  dplx
    *w;
  snew(w,ndim);
  snew(wmatrix,ndim*ndim);
  snew(V,ndim*ndim);
  snew(Vt,ndim*ndim);
  snew(temp,ndim*ndim);

  /* diagonalize */
  /* H shoudl be hermitian, so I hope it actually is... */
//  diag(ndim,w,V,A);
  diag_complex(ndim,w,V,A);

  for ( i = 0 ; i < ndim ; i++){
    wmatrix[i*ndim+i] = cexp((w[i]));
  }
  dagger(ndim,Vt,V);
  M_complextimesM_complex(ndim,wmatrix,V,temp);
  M_complextimesM_complex(ndim,Vt,temp,expA);

  free(Vt);
  free(V);
  free(w);
  free(wmatrix);
  free(temp);
}

static void expM_complex2(int ndim, dplx *A, dplx *expA,double dt){
  /* exp(-0.5*I*dt*A), equation B11 in JCP 114 10808 (2001) */
  int
    i,j;
  dplx
    *wmatrix,*V,*Vt,*temp;
  double
    *w;
  snew(w,ndim);
  snew(wmatrix,ndim*ndim);
  snew(V,ndim*ndim);
  snew(Vt,ndim*ndim);
  snew(temp,ndim*ndim);
  diag(ndim,w,V,A);
  for ( i = 0 ; i < ndim ; i++){
    wmatrix[i*ndim+i] = cexp((-0.5*IMAG*dt/AU2PS*w[i]));
  }
  dagger(ndim,Vt,V);
  M_complextimesM_complex(ndim,wmatrix,V,temp);
  M_complextimesM_complex(ndim,Vt,temp,expA);
  free(Vt);
  free(V);
  free(w);
  free(wmatrix);
  free(temp);
}

static void printM  ( int ndim, double *A){
  int 
    i,j;
  for(i=0;i<ndim;i++){
    for (j=0;j<ndim;j++){
      fprintf(stderr,"%lf ",A[i*ndim+j]);
    }
    fprintf(stderr,"\n\n");
  }
}

static void printM_complex  ( int ndim, dplx *A){
  int
    i,j;
  for(i=0;i<ndim;i++){
    for (j=0;j<ndim;j++){
      fprintf(stderr,"%lf+%lfI ",creal(A[i*ndim+j]),cimag(A[i*ndim+j]));
    }
    fprintf(stderr,"\n\n");
  }
}

/* integrate wavefunction for one MD timestep */
/* as before use the time-evolution operator.
 *  * relies on the the Intel MKL Lapack implementation.
 *   * this could be changed by adding some of the complex
 *    * Lapack routines to the Gromacs Lapack.
 *     */
static void  propagate_local_dia(int ndim,double dt, dplx *C, dplx *vec,
                           dplx *vecold, double *QMener, 
                           double *QMenerold,dplx *U){
  int
    i,j,k;
  double
    *E;
  dplx
    *S,*SS, *transposeS;
  dplx
    *invsqrtSS,*T,*transposeT,*ham;
  dplx
    *H,*expH,*ctemp;

  snew(S,ndim*ndim);
  snew(SS,ndim*ndim);
  snew(invsqrtSS,ndim*ndim);
  snew(transposeS,ndim*ndim);
  snew(T,ndim*ndim);
  snew(transposeT,ndim*ndim);
  snew(E,ndim*ndim);
  snew(ham,ndim*ndim);
  snew(H, ndim*ndim);
  snew(expH, ndim*ndim);
  snew(ctemp,ndim);

  for (i = 0 ; i < ndim ; i++){
    E[i*ndim+i]    = QMener[i];
    for ( j = 0 ; j < ndim ; j++ ){
      for ( k = 0 ; k < ndim ; k++ ){
	S[i*ndim+j] += conj(vecold[i*ndim+k])*vec[j*ndim+k];
      }
    }
  }
  /* build S^dagger S */
  transposeM_complex(ndim,S,transposeS);

  M_complextimesM_complex(ndim,transposeS,S,SS);

  invsqrtM_complex(ndim,SS,invsqrtSS);

  M_complextimesM_complex(ndim,S,invsqrtSS,T);

  transposeM_complex(ndim,T,transposeT);
  MtimesM_complex(ndim,E,transposeT,invsqrtSS);
 
  M_complextimesM_complex(ndim,T,invsqrtSS,ham);
 
  for ( i = 0 ; i< ndim; i++){
    ham[i+ndim*i] = creal(ham[i+ndim*i]) + QMenerold[i] + IMAG*cimag(ham[i+ndim*i]);
  }

  expM_complex2(ndim,ham,expH,dt);

  M_complextimesM_complex(ndim,transposeT,expH,U);
 
  /* we use U to propagate C, and keep it to compute the hopping
   * probabilities */
  MtimesV_complex(ndim,U,C,ctemp);
  for( i=0;i<ndim;i++){
    C[i]=ctemp[i];
  }

//
//  fprintf(stderr,"From propagate_local_dia, C matrix elements:\n");
//  for(i=0;i<ndim;i++){
//    fprintf(stderr,"C[%d]=%lf+%lfI ",i,creal(C[i]),cimag(C[i]));
//  }
//  fprintf(stderr,"\n");

  free(H);
  free(ham);
  free(T);
  free(transposeT);
  free(S);
  free(SS);
  free(invsqrtSS);
  free(transposeS);
  free(E);
  free(expH);
  free(ctemp);
} 

static void propagate(int dim, double dt, dplx *C, dplx *vec, dplx *vecold, double *QMener, double *QMenerold)
{
  int 
    i, j;
  int 
    n = dim;
  double 
    sum;
  double 
    *w;
  dplx 
    *H, *V, *Vt, *t;

/* build A matrix */
  snew(H, n*n);
  snew(w, n);
  snew(V, n*n);
  snew(Vt, n*n);
  snew(t, n);
  for (i=0; i<n; i++)
  {
    for (j=0; j<n; j++)
    {
      H[j + i*n] = -IMAG*calc_coupling_complex(j, i, dt/AU2PS, n, vec, vecold);
    }
    H[i + i*n] += (QMener[i] + QMenerold[i]) / 2.;
  }
// fprintf(stderr,"\n\nFrom propagate, EFFECTIVE HAMILTONIAN \n");
// printM_complex(n,H);
//fflush(stderr);

  diag(n, w, V, H);
// fprintf(stderr,"\n\n From propagate, EIGENVALUES \n");
// for (i=0; i<qm->nstates; i++){
//   fprintf(stderr,"%.5f\t ",w[i]);
// }       
// fprintf(stderr,"\n");               
// fflush(stderr);

// fprintf(stderr,"EIGENVECTORS \n");
// for (i=0; i<qm->nstates; i++){
//   for (j=0; j<qm->nstates; j++){
//     fprintf(stderr,"%.5f %.5f ",creal(V[j + i*n]),cimag(V[j + i*n]));
//   }
//   fprintf(stderr,"\n");
// }                      
// fflush(stderr);*/

  dagger(n, Vt, V);
  multmv(n, t, Vt, C);
  multexpdiag(n, C, -IMAG*dt/AU2PS, w, t);
  multmv(n, t, V, C);

  for (i=0; i<n; i++){
    C[i] = t[i];
  }

  sfree(H);
  sfree(w);
  sfree(V);
  sfree(Vt);
  sfree(t);
}

void eigensolver(real *a, int n, int index_lower, int index_upper, real *eigenvalues, real *eigenvectors){
  int 
    lwork,liwork,il,iu,m,iw0,info;
  int
    *isuppz,*iwork;
  real   
    w0,abstol,vl,vu;
  real
    *work;
  const char 
    *jobz;
//eigensolver(matrix,ndim,0,ndim,eigval,eigvec);    
  if(index_lower<0)
    index_lower = 0;
  
  if(index_upper>=n)
    index_upper = n-1;
    
  /* Make jobz point to the character "V" if eigenvectors
   * should be calculated, otherwise "N" (only eigenvalues).
   */   
  jobz = (eigenvectors != NULL) ? "V" : "N";

  /* allocate lapack stuff */
  snew(isuppz,2*n);
  vl = vu = 0;
    
  /* First time we ask the routine how much workspace it needs */
  lwork  = -1;
  liwork = -1;
  abstol = 0;
    
  /* Convert indices to fortran standard */
  index_lower++;
  index_upper++;
    
  /* Call LAPACK routine using fortran interface. Note that we use upper storage,
   * but this corresponds to lower storage ("L") in Fortran.
   */    
#ifdef GMX_DOUBLE
  F77_FUNC(dsyevr,DSYEVR)(jobz,"I","L",&n,a,&n,&vl,&vu,&index_lower,&index_upper,
                          &abstol,&m,eigenvalues,eigenvectors,&n,
                          isuppz,&w0,&lwork,&iw0,&liwork,&info);
#else
  F77_FUNC(ssyevr,SSYEVR)(jobz,"I","L",&n,a,&n,&vl,&vu,&index_lower,&index_upper,
                          &abstol,&m,eigenvalues,eigenvectors,&n,
                          isuppz,&w0,&lwork,&iw0,&liwork,&info);
#endif

  if(info != 0){
    sfree(isuppz);
    gmx_fatal(FARGS,"Internal errror in LAPACK diagonalization.");        
  }
    
  lwork = w0;
  liwork = iw0;
    
  snew(work,lwork);
  snew(iwork,liwork);
   
  abstol = 0;
    
#ifdef GMX_DOUBLE
  F77_FUNC(dsyevr,DSYEVR)(jobz,"I","L",&n,a,&n,&vl,&vu,&index_lower,&index_upper,
                          &abstol,&m,eigenvalues,eigenvectors,&n,
                          isuppz,work,&lwork,iwork,&liwork,&info);
#else
  F77_FUNC(ssyevr,SSYEVR)(jobz,"I","L",&n,a,&n,&vl,&vu,&index_lower,&index_upper,
                          &abstol,&m,eigenvalues,eigenvectors,&n,
                          isuppz,work,&lwork,iwork,&liwork,&info);
#endif
    
  sfree(isuppz);
  sfree(work);
  sfree(iwork);
    
  if(info != 0){
    gmx_fatal(FARGS,"Internal errror in LAPACK diagonalization.");
  }
}


#ifdef GMX_MPI_NOT
void sparse_parallel_eigensolver(gmx_sparsematrix_t *A, int neig, real *eigenvalues, real *eigenvectors, int maxiter){
  int
    iwork[80],iparam[11],ipntr[11];
  real 
    *resid,*workd,*workl,*v;
  int      
    n,ido,info,lworkl,i,ncv,dovec,iter,nnodes,rank;
  real
    abstol;
  int *
    select;

  MPI_Comm_size( MPI_COMM_WORLD, &nnodes );
  MPI_Comm_rank( MPI_COMM_WORLD, &rank );
	
  if(eigenvectors != NULL)
    dovec = 1;
  else
    dovec = 0;
    
  n   = A->nrow;
  ncv = 2*neig;
    
  if(ncv>n)
    ncv=n;
    
  for(i=0;i<11;i++)
        iparam[i]=ipntr[i]=0;
	
	iparam[0] = 1;       /* Don't use explicit shifts */
	iparam[2] = maxiter; /* Max number of iterations */
	iparam[6] = 1;       /* Standard symmetric eigenproblem */
    
	lworkl = ncv*(8+ncv);
    snew(resid,n);
    snew(workd,(3*n+4));
    snew(workl,lworkl);
    snew(select,ncv);
    snew(v,n*ncv);
	
    /* Use machine tolerance - roughly 1e-16 in double precision */
    abstol = 0;
    
 	ido = info = 0;
    fprintf(stderr,"Calculation Ritz values and Lanczos vectors, max %d iterations...\n",maxiter);
    
    iter = 1;
	do {
#ifdef GMX_DOUBLE
		F77_FUNC(pdsaupd,PDSAUPD)(&ido, "I", &n, "SA", &neig, &abstol, 
								  resid, &ncv, v, &n, iparam, ipntr, 
								  workd, iwork, workl, &lworkl, &info);
#else
		F77_FUNC(pssaupd,PSSAUPD)(&ido, "I", &n, "SA", &neig, &abstol, 
								  resid, &ncv, v, &n, iparam, ipntr, 
								  workd, iwork, workl, &lworkl, &info);
#endif
        if(ido==-1 || ido==1)
            gmx_sparsematrix_vector_multiply(A,workd+ipntr[0]-1, workd+ipntr[1]-1);
        
        fprintf(stderr,"\rIteration %4d: %3d out of %3d Ritz values converged.",iter++,iparam[4],neig);
	} while(info==0 && (ido==-1 || ido==1));
	
    fprintf(stderr,"\n");
	if(info==1){
	    gmx_fatal(FARGS,
                  "Maximum number of iterations (%d) reached in Arnoldi\n"
                  "diagonalization, but only %d of %d eigenvectors converged.\n",
                  maxiter,iparam[4],neig);
	}
	else if(info!=0){
        gmx_fatal(FARGS,"Unspecified error from Arnoldi diagonalization:%d\n",info);
	}
	
	info = 0;
	/* Extract eigenvalues and vectors from data */
    fprintf(stderr,"Calculating eigenvalues and eigenvectors...\n");
    
#ifdef GMX_DOUBLE
    F77_FUNC(pdseupd,PDSEUPD)(&dovec, "A", select, eigenvalues, eigenvectors, 
							  &n, NULL, "I", &n, "SA", &neig, &abstol, 
							  resid, &ncv, v, &n, iparam, ipntr, 
							  workd, workl, &lworkl, &info);
#else
    F77_FUNC(psseupd,PSSEUPD)(&dovec, "A", select, eigenvalues, eigenvectors, 
							  &n, NULL, "I", &n, "SA", &neig, &abstol, 
							  resid, &ncv, v, &n, iparam, ipntr, 
							  workd, workl, &lworkl, &info);
#endif
	
    sfree(v);
    sfree(resid);
    sfree(workd);
    sfree(workl);  
    sfree(select);    
}
#endif


void 
sparse_eigensolver(gmx_sparsematrix_t *    A,
                   int                     neig,
                   real *                  eigenvalues,
                   real *                  eigenvectors,
                   int                     maxiter)
{
    int      iwork[80];
    int      iparam[11];
    int      ipntr[11];
    real *   resid;
    real *   workd;
    real *   workl;
    real *   v;
    int      n;
    int      ido,info,lworkl,i,ncv,dovec;
    real     abstol;
    int *    select;
    int      iter;
    
#ifdef GMX_MPI_NOT
	MPI_Comm_size( MPI_COMM_WORLD, &n );
	if(n > 1)
	{
		sparse_parallel_eigensolver(A,neig,eigenvalues,eigenvectors,maxiter);
		return;
	}
#endif
	
    if(eigenvectors != NULL)
        dovec = 1;
    else
        dovec = 0;
    
    n   = A->nrow;
    ncv = 2*neig;
    
    if(ncv>n)
        ncv=n;
    
    for(i=0;i<11;i++)
        iparam[i]=ipntr[i]=0;
	
	iparam[0] = 1;       /* Don't use explicit shifts */
	iparam[2] = maxiter; /* Max number of iterations */
	iparam[6] = 1;       /* Standard symmetric eigenproblem */
    
	lworkl = ncv*(8+ncv);
    snew(resid,n);
    snew(workd,(3*n+4));
    snew(workl,lworkl);
    snew(select,ncv);
    snew(v,n*ncv);

    /* Use machine tolerance - roughly 1e-16 in double precision */
    abstol = 0;
    
 	ido = info = 0;
    fprintf(stderr,"Calculation Ritz values and Lanczos vectors, max %d iterations...\n",maxiter);
    
    iter = 1;
	do {
#ifdef GMX_DOUBLE
            F77_FUNC(dsaupd,DSAUPD)(&ido, "I", &n, "SA", &neig, &abstol, 
                                    resid, &ncv, v, &n, iparam, ipntr, 
                                    workd, iwork, workl, &lworkl, &info);
#else
            F77_FUNC(ssaupd,SSAUPD)(&ido, "I", &n, "SA", &neig, &abstol, 
                                    resid, &ncv, v, &n, iparam, ipntr, 
                                    workd, iwork, workl, &lworkl, &info);
#endif
        if(ido==-1 || ido==1)
            gmx_sparsematrix_vector_multiply(A,workd+ipntr[0]-1, workd+ipntr[1]-1);
        
        fprintf(stderr,"\rIteration %4d: %3d out of %3d Ritz values converged.",iter++,iparam[4],neig);
	} while(info==0 && (ido==-1 || ido==1));
	
    fprintf(stderr,"\n");
	if(info==1)
    {
	    gmx_fatal(FARGS,
                  "Maximum number of iterations (%d) reached in Arnoldi\n"
                  "diagonalization, but only %d of %d eigenvectors converged.\n",
                  maxiter,iparam[4],neig);
    }
	else if(info!=0)
    {
        gmx_fatal(FARGS,"Unspecified error from Arnoldi diagonalization:%d\n",info);
    }
	
	info = 0;
	/* Extract eigenvalues and vectors from data */
    fprintf(stderr,"Calculating eigenvalues and eigenvectors...\n");
    
#ifdef GMX_DOUBLE
    F77_FUNC(dseupd,DSEUPD)(&dovec, "A", select, eigenvalues, eigenvectors, 
			    &n, NULL, "I", &n, "SA", &neig, &abstol, 
			    resid, &ncv, v, &n, iparam, ipntr, 
			    workd, workl, &lworkl, &info);
#else
    F77_FUNC(sseupd,SSEUPD)(&dovec, "A", select, eigenvalues, eigenvectors, 
			    &n, NULL, "I", &n, "SA", &neig, &abstol, 
			    resid, &ncv, v, &n, iparam, ipntr, 
			    workd, workl, &lworkl, &info);
#endif
	
    sfree(v);
    sfree(resid);
    sfree(workd);
    sfree(workl);  
    sfree(select);    
}
/* end of eigensolver code */


/* TODO: this should be made thread-safe */

/* Gaussian interface routines */
void init_gaussian(t_commrec *cr, t_QMrec *qm, t_MMrec *mm){
  FILE    
    *rffile=NULL,*out=NULL,*Cin=NULL,*zin=NULL,*eigvecin=NULL;
  ivec
    basissets[eQMbasisNR]={{0,3,0},
			   {0,3,0},/*added for double sto-3g entry in names.c*/
			   {5,0,0},
			   {5,0,1},
			   {5,0,11},
			   {5,6,0},
			   {1,6,0},
			   {1,6,1},
			   {1,6,11},
			   {4,6,0}};
  char
    *buf;
  int
    i,j,ndim=1,seed,print=1;
  double
    *eig_real,*eig_imag; 
  dplx
    *eig;
 
  /* using the ivec above to convert the basis read form the mdp file
   * in a human readable format into some numbers for the gaussian
   * route. This is necessary as we are using non standard routes to
   * do SH.
   */

  /* per layer we make a new subdir for integral file, checkpoint
   * files and such. These dirs are stored in the QMrec for
   * convenience 
   */

  
  if(!qm->nQMcpus){ /* this we do only once per layer 
		     * as we call g01 externally 
		     */

    for(i=0;i<DIM;i++)
      qm->SHbasis[i]=basissets[qm->QMbasis][i];

  /* init gradually switching on of the SA */
      qm->SAstep = 0;
  /* we read the number of cpus and environment from the environment
   * if set.  
   */
      snew(buf,20);
      buf = getenv("NCPUS");
      if (buf)
        sscanf(buf,"%d",&qm->nQMcpus);
      else
        qm->nQMcpus=1;
      fprintf(stderr,"number of CPUs for gaussian = %d\n",qm->nQMcpus);
      snew(buf,50);
      buf = getenv("MEM");
      if (buf)
        sscanf(buf,"%d",&qm->QMmem);
      else
        qm->QMmem=50000000;
      fprintf(stderr,"memory for gaussian = %d\n",qm->QMmem);
      snew(buf,30);
      buf = getenv("ACC");
    if (buf)
      sscanf(buf,"%d",&qm->accuracy);
    else
      qm->accuracy=8;  
    fprintf(stderr,"accuracy in l510 = %d\n",qm->accuracy); 
    snew(buf,30);
    buf = getenv("CPMCSCF");
    if (buf)
	{
		sscanf(buf,"%d",&i);
		qm->cpmcscf = (i!=0);
	}
	else
      qm->cpmcscf=FALSE;
    if (qm->cpmcscf)
      fprintf(stderr,"using cp-mcscf in l1003\n");
    else
      fprintf(stderr,"NOT using cp-mcscf in l1003\n"); 
    snew(buf,50);
    buf = getenv("SASTEP");
    if (buf)
      sscanf(buf,"%d",&qm->SAstep);
    else
      /* init gradually switching on of the SA */
      qm->SAstep = 0;
    /* we read the number of cpus and environment from the environment
     * if set.  
     */
    fprintf(stderr,"Level of SA at start = %d\n",qm->SAstep);
        

    /* punch the LJ C6 and C12 coefficients to be picked up by
     * gaussian and usd to compute the LJ interaction between the
     * MM and QM atoms.
     */
    if(qm->bTS||qm->bOPT){
      out = fopen("LJ.dat","w");
      for(i=0;i<qm->nrQMatoms;i++){

#ifdef GMX_DOUBLE
	fprintf(out,"%3d  %10.7lf  %10.7lf\n",
		qm->atomicnumberQM[i],qm->c6[i],qm->c12[i]);
#else
	fprintf(out,"%3d  %10.7f  %10.7f\n",
		qm->atomicnumberQM[i],qm->c6[i],qm->c12[i]);
#endif
      }
      fclose(out);
    }
    /* gaussian settings on the system */
    snew(buf,200);
    buf = getenv("GAUSS_DIR");

    if (buf){
      snew(qm->gauss_dir,200);
      sscanf(buf,"%s",qm->gauss_dir);
    }
    else
      gmx_fatal(FARGS,"no $GAUSS_DIR, check gaussian manual\n");
    
    snew(buf,200);    
    buf = getenv("GAUSS_EXE");
    if (buf){
      snew(qm->gauss_exe,200);
      sscanf(buf,"%s",qm->gauss_exe);
    }
    else
      gmx_fatal(FARGS,"no $GAUSS_EXE, check gaussian manual\n");
    
    snew(buf,200);
    buf = getenv("DEVEL_DIR");
    if (buf){
      snew(qm->devel_dir,200);
      sscanf(buf,"%s",qm->devel_dir);
    }
    else
      gmx_fatal(FARGS,"no $DEVEL_DIR, this is were the modified links reside.\n");


    if(qm->bQED){
      fprintf(stderr,"\nDoing QED");
      /* prepare for a cavity QED MD run. Obviously only works with QM/MM */
      if(MULTISIM(cr)){
	fprintf(stderr,"doing parallel; ms->nsim, ms_>sim = %d,%d\n", 
                cr->ms->nsim,cr->ms->sim);
	/* Un-setting boolean for printing for nodes != node0 */   
	if (cr->ms->sim!=0){
	  print=0;
	}
	snew(buf,3000);
        buf = getenv("TMP_DIR");
        if (buf){
          snew(qm->subdir,3000);
          /* store the nodeid as the subdir */
          sprintf(qm->subdir,"%s%s%d",buf,"/molecule",cr->ms->sim);
          /* and create the directoru on the FS */
          sprintf(buf,"%s %s","mkdir",qm->subdir);
          system(buf);
          ndim=cr->ms->nsim;
        }
        else
          gmx_fatal(FARGS,"no $TMP_DIR, this is were the temporary in/output is written.\n");
      }
      ndim+=(qm->n_max-qm->n_min+1);
      snew(qm->creal,ndim);
      snew(qm->cimag,ndim);
      Cin=fopen("C.dat","r");
      if (Cin){
        fprintf(stderr,"reading coefficients from C.dat\n");
        if(NULL == fgets(buf,3000,Cin)){
          gmx_fatal(FARGS,"Error reading C.dat");
        }
        sscanf(buf,"%d\n",&seed);
        fprintf(stderr,"setting randon seed to %d\n",seed);
        for(i=0;i<ndim;i++){
          if(NULL == fgets(buf,3000,Cin)){
            gmx_fatal(FARGS,"Error reading C.dat, no expansion coeficient");
          }
          sscanf(buf,"%lf %lf",&qm->creal[i],&qm->cimag[i]);
        }
        /* rho 0, only population */
        if(NULL == fgets(buf,3000,Cin)){
          gmx_fatal(FARGS,"Error reading C.dat: no rho0");
        }
        sscanf(buf,"%lf",&qm->groundstate);

        fclose(Cin);

	/* print for security */   
        if(print){ 
          fprintf (stderr,"coefficients\nC: ");
          for(i=0;i<ndim;i++){
            fprintf(stderr,"%lf ",conj(qm->creal[i]+IMAG*qm->cimag[i])*(qm->creal[i]+IMAG*qm->cimag[i]));
          }
          fprintf(stderr,"rho0: %lf ",qm->groundstate);
        }
      }
      else {
        snew(buf,200);
        buf = getenv("SEED");
        if (buf){
          sscanf(buf,"%d",&seed);
        }
        fprintf(stderr,"No C.dat file, setting C[%d]=1.0+0.0I\n",qm->polariton);
        qm->creal[qm->polariton]=1;
        qm->groundstate=0.0;
        fprintf(stderr,"setting randon seed to %d\n",seed);
      }

      /* File for positioning the molecules as desired along the z axis */
      snew(qm->z,ndim-(qm->n_max-qm->n_min+1));
      zin=fopen("z.dat","r");
      if(zin){
	fprintf(stderr,"z.dat file exists, reading z-positions\n");
	for(i=0;i<ndim-(qm->n_max-qm->n_min+1);i++){
	  if(NULL == fgets(buf,300,zin)){
	    gmx_fatal(FARGS,"Error reading z.dat, check its content\n");
	  }
	  else{
	    sscanf(buf,"%lf",&qm->z[i]);
	    qm->z[i]*=microM2BOHR;
	    if (MULTISIM(cr)){
	      if (cr->ms->sim==0){
	        fprintf(stderr,"node %d, z[%d]=%lf\n",cr->ms->sim,i,qm->z[i]);
	      }
	    }
	    else{
 	      fprintf(stderr,"z[%d]=%lf\n",i,qm->z[i]);
	    }
	  }
	}
	fclose(zin);
      }
      else{
	if(print){
	  fprintf(stderr,"No z-positions file, setting molecules equally spaced along the z-axis\n");
	}
	for(i=0;i<ndim-(qm->n_max-qm->n_min+1);i++){
	  qm->z[i]=i*qm->L*microM2BOHR/(ndim-(qm->n_max-qm->n_min+1));
	}
      }

      /* File for providing one step eigenvectors and avoid random phases in continuation runs */

      /* eigvecin if a file cointaining the eigenvectors for one MD step. It has ndim*ndim lines, 
	 each line cointaining two long floats (the real and imaginary parts) of the ndim components 
	 of each eigenvector. For now, use the following for preparing this files:
	 tail -n $ndim eigenvectors.dat | awk '{$1=$2=$3=$4=$5=$6=$7=$8=$9=$10=$11=""; print $0}' | sed 's/\+//g' |
	 sed 's/I/\n/g' | sed '/^$/d' | sed 's/          //g' > eig_vec_last.dat */

      if(qm->bContinuation){
	eigvecin=fopen("eig_vec_last.dat","r");
	if(eigvecin){
	  fprintf(stderr,"eig_vec_last.dat file exists, reading eigenvectors from last step of previous run\n");
	  snew(eig_real,ndim*ndim);
	  snew(eig_imag,ndim*ndim);
	  for(i=0;i<ndim*ndim;i++){
	    if(NULL == fgets(buf,2*sizeof(double)+4,eigvecin)){
	      gmx_fatal(FARGS,"Error reading eig_vec_last.dat, check its content\n");
	    }
	    else{
	      sscanf(buf,"%lf %lf",&eig_real[i],&eig_imag[i]);
	    }
	  };
	  snew(eig,ndim*ndim);
	  for(i=0;i<ndim*ndim;i++){
	    eig[i]=eig_real[i]+IMAG*eig_imag[i];
	  }
	  /* copy the eigenvectors read from eig_vec_last.dat to qmrec */
	  snew(qm->eigvec,ndim*ndim);
	  for(i=0;i<ndim*ndim;i++){
	    qm->eigvec[i]=eig[i];
	  };
          /* Print to check correcteness of read */
	  if(print){
	    fprintf(stderr,"node %d, Eigenvectors previous step:\n",cr->ms->sim);
	    for(i=0;i<ndim;i++){
	      fprintf(stderr,"Eig[%d]= ",i);
	      for(j=0;j<ndim;j++){
		fprintf(stderr,"%lf + %lf I ",creal(qm->eigvec[i*ndim+j]),cimag(qm->eigvec[i*ndim+j]));
	      };
	      fprintf(stderr,"\n");
	    };
	  }
	  /* Set qm boolean to indicate it is a Continuation run */
	  qm->QEDrestart=1;
	  fclose(eigvecin);
	}
	else{
	  if(print){
	    fprintf(stderr,"Warning: No eigenvectors file for last step of previous run, eigenvectors phases will be undetermined\n");
	  }
  	  snew(qm->eigvec,ndim*ndim);
	  qm->QEDrestart=0;
	}
      }
      else{
	snew(qm->eigvec,ndim*ndim);
	qm->QEDrestart=0;
      }

      snew(qm->rnr,qm->nsteps);
      srand(seed);
      for (i=0;i< qm->nsteps;i++){
        qm->rnr[i]=(double) rand()/(RAND_MAX*1.0);
      }
      snew(qm->eigval,ndim);
      snew(buf,3000);
      buf = getenv("WORK_DIR");
      if (buf){
        snew(qm->work_dir,3000);
        sscanf(buf,"%s",qm->work_dir);
      }
      else
        gmx_fatal(FARGS,"no $WORK_DIR, this is were the QED-specific output is written.\n");
    }
  }
  fprintf(stderr,"gaussian initialised...\n");
}  


void write_gaussian_SH_input(int step, gmx_bool swap, t_forcerec *fr, t_QMrec *qm, t_MMrec *mm){
  int
    i;
  gmx_bool
    bSA;
  FILE
    *out;
  t_QMMMrec
    *QMMMrec;

  QMMMrec = fr->qr;
  bSA = (qm->SAstep>0);

  out = fopen("input.com","w");
  /* write the route */
  fprintf(out,"%s","%scr=input\n");
  fprintf(out,"%s","%rwf=input\n");
  fprintf(out,"%s","%int=input\n");
  fprintf(out,"%s","%d2e=input\n");
/*  if(step)
 *   fprintf(out,"%s","%nosave\n");
 */
  fprintf(out,"%s","%chk=input\n");
  fprintf(out,"%s%d\n","%mem=",qm->QMmem);
  fprintf(out,"%s%3d\n","%nprocshare=",qm->nQMcpus);

  /* use the versions of
   * l301 that computes the interaction between MM and QM atoms.
   * l510 that can punch the CI coefficients
   * l701 that can do gradients on MM atoms 
   */

  /* local version */
  fprintf(out,"%s%s%s",
	  "%subst l510 ",
	  qm->devel_dir,
	  "/l510\n");
  fprintf(out,"%s%s%s",
	  "%subst l301 ",
	  qm->devel_dir,
	  "/l301\n");
  fprintf(out,"%s%s%s",
	  "%subst l701 ",
	  qm->devel_dir,
	  "/l701\n");
  
  fprintf(out,"%s%s%s",
	  "%subst l1003 ",
	  qm->devel_dir,
	  "/l1003\n");
  fprintf(out,"%s%s%s",
	  "%subst l9999 ",
	  qm->devel_dir,
	  "/l9999\n");
  /* print the nonstandard route 
   */
  fprintf(out,"%s",
	  "#P nonstd\n 1/18=10,20=1,38=1/1;\n");
  fprintf(out,"%s",
	  " 2/9=110,15=1,17=6,18=5,40=1/2;\n");
  if(mm->nrMMatoms)
    fprintf(out,
	    " 3/5=%d,6=%d,7=%d,25=1,32=1,43=1,94=-2/1,2,3;\n",
	    qm->SHbasis[0],
	    qm->SHbasis[1],
	    qm->SHbasis[2]); /*basisset stuff */
  else
    fprintf(out,
	    " 3/5=%d,6=%d,7=%d,25=1,32=1,43=0,94=-2/1,2,3;\n",
	    qm->SHbasis[0],
	    qm->SHbasis[1],
	    qm->SHbasis[2]); /*basisset stuff */
  /* development */
  if (step+1) /* fetch initial guess from check point file */
    /* hack, to alyays read from chk file!!!!! */
    fprintf(out,"%s%d,%s%d%s"," 4/5=1,7=6,17=",
	    qm->CASelectrons,
	    "18=",qm->CASorbitals,"/1,5;\n");
  else /* generate the first checkpoint file */
    fprintf(out,"%s%d,%s%d%s"," 4/5=0,7=6,17=",
	    qm->CASelectrons,
	    "18=",qm->CASorbitals,"/1,5;\n");
  /* the rest of the input depends on where the system is on the PES 
   */
  if(swap && bSA){ /* make a slide to the other surface */
    if(qm->CASorbitals>8){  /* use direct and no full diag */
      fprintf(out," 5/5=2,7=512,16=-2,17=10000000,28=2,32=2,38=6,97=100/10;\n");
    } 
    else {
      if(qm->cpmcscf){
	fprintf(out," 5/5=2,6=%d,7=512,17=31000200,28=2,32=2,38=6,97=100/10;\n",
		qm->accuracy);
	if(mm->nrMMatoms>0)
	  fprintf(out," 7/7=1,16=-2,30=1/1;\n");
	fprintf(out," 11/31=1,42=1,45=1/1;\n");
	fprintf(out," 10/6=1,10=700006,28=2,29=1,31=1,97=100/3;\n");
	fprintf(out," 7/30=1/16;\n 99/10=4/99;\n");
      }
      else{
	fprintf(out," 5/5=2,6=%d,7=512,17=11000000,28=2,32=2,38=6,97=100/10;\n",
		qm->accuracy);
	fprintf(out," 7/7=1,16=-2,30=1/1,2,3,16;\n 99/10=4/99;\n");
      }
    }
  }
  else if(bSA){ /* do a "state-averaged" CAS calculation */
    if(qm->CASorbitals>8){ /* no full diag */ 
      fprintf(out," 5/5=2,7=512,16=-2,17=10000000,28=2,32=2,38=6/10;\n");
    } 
    else {
      if(qm->cpmcscf){
	fprintf(out," 5/5=2,6=%d,7=512,17=31000200,28=2,32=2,38=6/10;\n",
		qm->accuracy);
	if(mm->nrMMatoms>0)
	  fprintf(out," 7/7=1,16=-2,30=1/1;\n");
	fprintf(out," 11/31=1,42=1,45=1/1;\n");
	fprintf(out," 10/6=1,10=700006,28=2,29=1,31=1/3;\n");
	fprintf(out," 7/30=1/16;\n 99/10=4/99;\n");
      }
      else{
      	fprintf(out," 5/5=2,6=%d,7=512,17=11000000,28=2,32=2,38=6/10;\n",
		qm->accuracy);
	fprintf(out," 7/7=1,16=-2,30=1/1,2,3,16;\n 99/10=4/99;\n");
      }
    }
  }
  else if(swap){/* do a "swapped" CAS calculation */
    if(qm->CASorbitals>8)
      fprintf(out," 5/5=2,7=512,16=-2,17=0,28=2,32=2,38=6,97=100/10;\n");
    else
      fprintf(out," 5/5=2,6=%d,7=512,17=1000000,28=2,32=2,38=6,97=100/10;\n",
	      qm->accuracy);
    fprintf(out," 7/7=1,16=-2,30=1/1,2,3,16;\n 99/10=4/99;\n");
  }
  else {/* do a "normal" CAS calculation */
    if(qm->CASorbitals>8)
      fprintf(out," 5/5=2,7=512,16=-2,17=0,28=2,32=2,38=6/10;\n");
    else
      fprintf(out," 5/5=2,6=%d,7=512,17=1000000,28=2,32=2,38=6/10;\n",
	      qm->accuracy);
    fprintf(out," 7/7=1,16=-2,30=1/1,2,3,16;\n 99/10=4/99;\n");
  }
  fprintf(out, "\ninput-file generated by gromacs\n\n");
  fprintf(out,"%2d%2d\n",qm->QMcharge,qm->multiplicity);
  for (i=0;i<qm->nrQMatoms;i++){
#ifdef GMX_DOUBLE
    fprintf(out,"%3d %10.7lf  %10.7lf  %10.7lf\n",
	    qm->atomicnumberQM[i],
	    qm->xQM[i][XX]/BOHR2NM,
	    qm->xQM[i][YY]/BOHR2NM,
	    qm->xQM[i][ZZ]/BOHR2NM);
#else
    fprintf(out,"%3d %10.7f  %10.7f  %10.7f\n",
	    qm->atomicnumberQM[i],
	    qm->xQM[i][XX]/BOHR2NM,
	    qm->xQM[i][YY]/BOHR2NM,
	    qm->xQM[i][ZZ]/BOHR2NM);
#endif
  }
  /* MM point charge data */
  if(QMMMrec->QMMMscheme!=eQMMMschemeoniom && mm->nrMMatoms){
    fprintf(out,"\n");
    for(i=0;i<mm->nrMMatoms;i++){
#ifdef GMX_DOUBLE
      fprintf(out,"%10.7lf  %10.7lf  %10.7lf %8.4lf\n",
	      mm->xMM[i][XX]/BOHR2NM,
	      mm->xMM[i][YY]/BOHR2NM,
	      mm->xMM[i][ZZ]/BOHR2NM,
	      mm->MMcharges[i]);
#else
      fprintf(out,"%10.7f  %10.7f  %10.7f %8.4f\n",
	      mm->xMM[i][XX]/BOHR2NM,
	      mm->xMM[i][YY]/BOHR2NM,
	      mm->xMM[i][ZZ]/BOHR2NM,
	      mm->MMcharges[i]);
#endif
    }
  }
  if(bSA) {/* put the SA coefficients at the end of the file */
#ifdef GMX_DOUBLE
    fprintf(out,"\n%10.8lf %10.8lf\n",
	    qm->SAstep*0.5/qm->SAsteps,
	    1-qm->SAstep*0.5/qm->SAsteps);
#else    
    fprintf(out,"\n%10.8f %10.8f\n",
	    qm->SAstep*0.5/qm->SAsteps,
	    1-qm->SAstep*0.5/qm->SAsteps);
#endif
    fprintf(stderr,"State Averaging level = %d/%d\n",qm->SAstep,qm->SAsteps);
  }
  fprintf(out,"\n");
  fclose(out);
}  /* write_gaussian_SH_input */

void write_gaussian_input(int step ,t_forcerec *fr, t_QMrec *qm, t_MMrec *mm)
{
  int
    i;
  t_QMMMrec
    *QMMMrec;
  FILE
    *out;
  
  QMMMrec = fr->qr;
  out = fopen("input.com","w");
  /* write the route */

  if(qm->QMmethod>=eQMmethodRHF)
    fprintf(out,"%s",
	    "%chk=input\n");
  else
    fprintf(out,"%s",
	    "%chk=se\n");
  if(qm->nQMcpus>1)
    fprintf(out,"%s%3d\n",
	    "%nprocshare=",qm->nQMcpus);
  fprintf(out,"%s%d\n",
	  "%mem=",qm->QMmem);
  /* use the modified links that include the LJ contribution at the QM level */
  if(qm->bTS||qm->bOPT){
    fprintf(out,"%s%s%s",
	    "%subst l701 ",qm->devel_dir,"/l701_LJ\n");
    fprintf(out,"%s%s%s",
	    "%subst l301 ",qm->devel_dir,"/l301_LJ\n");
  }
  else{
    fprintf(out,"%s%s%s",
	    "%subst l701 ",qm->devel_dir,"/l701\n");
    fprintf(out,"%s%s%s",
	    "%subst l301 ",qm->devel_dir,"/l301\n");
  }
  fprintf(out,"%s%s%s",
	  "%subst l9999 ",qm->devel_dir,"/l9999\n");
  if(step){
    fprintf(out,"%s",
	    "#T ");
  }else{
    fprintf(out,"%s",
	    "#P ");
  }
  if(qm->QMmethod==eQMmethodB3LYPLAN){
    fprintf(out," %s", 
	    "B3LYP/GEN Pseudo=Read");
  }
  else{
    fprintf(out," %s", 
	    eQMmethod_names[qm->QMmethod]);
    
    if(qm->QMmethod>=eQMmethodRHF){
      fprintf(out,"/%s",
	      eQMbasis_names[qm->QMbasis]);
      if(qm->QMmethod==eQMmethodCASSCF){
	/* in case of cas, how many electrons and orbitals do we need?
	 */
	fprintf(out,"(%d,%d)",
		qm->CASelectrons,qm->CASorbitals);
      }
    }
  }
  if(QMMMrec->QMMMscheme==eQMMMschemenormal){
    fprintf(out," %s",
	    "Charge ");
  }
  if (step || qm->QMmethod==eQMmethodCASSCF){
    /* fetch guess from checkpoint file, always for CASSCF */
    fprintf(out,"%s"," guess=read");
  }
  fprintf(out,"\nNosymm units=bohr\n");
  
  if(qm->bTS){
    fprintf(out,"OPT=(Redundant,TS,noeigentest,ModRedundant) Punch=(Coord,Derivatives) ");
  }
  else if (qm->bOPT){
    fprintf(out,"OPT=(Redundant,ModRedundant) Punch=(Coord,Derivatives) ");
  }
  else{
    fprintf(out,"FORCE Punch=(Derivatives) ");
  }
  fprintf(out,"iop(3/33=1)\n\n");
  fprintf(out, "input-file generated by gromacs\n\n");
  fprintf(out,"%2d%2d\n",qm->QMcharge,qm->multiplicity);
  for (i=0;i<qm->nrQMatoms;i++){
#ifdef GMX_DOUBLE
    fprintf(out,"%3d %10.7lf  %10.7lf  %10.7lf\n",
	    qm->atomicnumberQM[i],
	    qm->xQM[i][XX]/BOHR2NM,
	    qm->xQM[i][YY]/BOHR2NM,
	    qm->xQM[i][ZZ]/BOHR2NM);
#else
    fprintf(out,"%3d %10.7f  %10.7f  %10.7f\n",
	    qm->atomicnumberQM[i],
	    qm->xQM[i][XX]/BOHR2NM,
	    qm->xQM[i][YY]/BOHR2NM,
	    qm->xQM[i][ZZ]/BOHR2NM);
#endif
  }

  /* Pseudo Potential and ECP are included here if selected (MEthod suffix LAN) */
  if(qm->QMmethod==eQMmethodB3LYPLAN){
    fprintf(out,"\n");
    for(i=0;i<qm->nrQMatoms;i++){
      if(qm->atomicnumberQM[i]<21){
	fprintf(out,"%d ",i+1);
      }
    }
    fprintf(out,"\n%s\n****\n",eQMbasis_names[qm->QMbasis]);
    
    for(i=0;i<qm->nrQMatoms;i++){
      if(qm->atomicnumberQM[i]>21){
	fprintf(out,"%d ",i+1);
      }
    }
    fprintf(out,"\n%s\n****\n\n","lanl2dz");    
    
    for(i=0;i<qm->nrQMatoms;i++){
      if(qm->atomicnumberQM[i]>21){
	fprintf(out,"%d ",i+1);
      }
    }
    fprintf(out,"\n%s\n","lanl2dz");    
  }    
    
  /* MM point charge data */
  if(QMMMrec->QMMMscheme!=eQMMMschemeoniom && mm->nrMMatoms){
    fprintf(stderr,"nr mm atoms in gaussian.c = %d\n",mm->nrMMatoms);
    fprintf(out,"\n");
    if(qm->bTS||qm->bOPT){
      /* freeze the frontier QM atoms and Link atoms. This is
       * important only if a full QM subsystem optimization is done
       * with a frozen MM environmeent. For dynamics, or gromacs's own
       * optimization routines this is not important.
       */
      for(i=0;i<qm->nrQMatoms;i++){
	if(qm->frontatoms[i]){
	  fprintf(out,"%d F\n",i+1); /* counting from 1 */
	}
      }
      /* MM point charges include LJ parameters in case of QM optimization
       */
      for(i=0;i<mm->nrMMatoms;i++){
#ifdef GMX_DOUBLE
	fprintf(out,"%10.7lf  %10.7lf  %10.7lf %8.4lf 0.0 %10.7lf %10.7lf\n",
		mm->xMM[i][XX]/BOHR2NM,
		mm->xMM[i][YY]/BOHR2NM,
		mm->xMM[i][ZZ]/BOHR2NM,
		mm->MMcharges[i],
		mm->c6[i],mm->c12[i]);
#else
	fprintf(out,"%10.7f  %10.7f  %10.7f %8.4f 0.0 %10.7f %10.7f\n",
		mm->xMM[i][XX]/BOHR2NM,
		mm->xMM[i][YY]/BOHR2NM,
		mm->xMM[i][ZZ]/BOHR2NM,
		mm->MMcharges[i],
		mm->c6[i],mm->c12[i]);
#endif
      }
      fprintf(out,"\n");
    }
    else{
      for(i=0;i<mm->nrMMatoms;i++){
#ifdef GMX_DOUBLE
	fprintf(out,"%10.7lf  %10.7lf  %10.7lf %8.4lf\n",
		mm->xMM[i][XX]/BOHR2NM,
		mm->xMM[i][YY]/BOHR2NM,
		mm->xMM[i][ZZ]/BOHR2NM,
		mm->MMcharges[i]);
#else
	fprintf(out,"%10.7f  %10.7f  %10.7f %8.4f\n",
		mm->xMM[i][XX]/BOHR2NM,
		mm->xMM[i][YY]/BOHR2NM,
		mm->xMM[i][ZZ]/BOHR2NM,
		mm->MMcharges[i]);
#endif
      }
    }
  }
  fprintf(out,"\n");
  

  fclose(out);

}  /* write_gaussian_input */


void write_gaussian_input_QED(t_commrec *cr,int step ,t_forcerec *fr, t_QMrec *qm, t_MMrec *mm){
  int
    i;
  t_QMMMrec
    *QMMMrec;
  FILE
    *out;
  
  QMMMrec = fr->qr;

  /* move to a new working directory! */
  if(MULTISIM(cr)){
    chdir (qm->subdir);
  }
  out = fopen("input.com","w");
  /* write the route */

  if(qm->QMmethod>=eQMmethodRHF)
    fprintf(out,"%s",
	    "%chk=input\n");
  else
    fprintf(out,"%s",
	    "%chk=se\n");
  if(qm->nQMcpus>1)
    fprintf(out,"%s%3d\n",
	    "%nprocshare=",qm->nQMcpus);
  fprintf(out,"%s%d\n",
	  "%mem=",qm->QMmem);
  /* use the modified links that include the LJ contribution at the QM level */
  if(qm->bTS||qm->bOPT){
    fprintf(out,"%s%s%s",
	    "%subst l701 ",qm->devel_dir,"/l701_LJ\n");
    fprintf(out,"%s%s%s",
	    "%subst l301 ",qm->devel_dir,"/l301_LJ\n");
  }
  else{
    fprintf(out,"%s%s%s",
	    "%subst l701 ",qm->devel_dir,"/l701\n");
    fprintf(out,"%s%s%s",
	    "%subst l301 ",qm->devel_dir,"/l301\n");
  }
  fprintf(out,"%s%s%s",
	  "%subst l9999 ",qm->devel_dir,"/l9999\n");
  if(step){
    fprintf(out,"%s",
	    "#T ");
  }else{
/* MOD 12.11.20116 */
    fprintf(out,"%s",
	    "#T ");
  }
  if(qm->QMmethod==eQMmethodB3LYPLAN){
    fprintf(out," %s", 
	    "B3LYP/GEN Pseudo=Read");
  }
  else{
    fprintf(out," %s", 
	    eQMmethod_names[qm->QMmethod]);
    
    if(qm->QMmethod>=eQMmethodRHF){
      fprintf(out,"/%s",
	      eQMbasis_names[qm->QMbasis]);
      if(qm->QMmethod==eQMmethodCASSCF){
	/* in case of cas, how many electrons and orbitals do we need?
	 */
	fprintf(out,"(%d,%d)",
		qm->CASelectrons,qm->CASorbitals);
      }
    }
  }
  if(QMMMrec->QMMMscheme==eQMMMschemenormal){
    fprintf(out," %s",
	    "Charge ");
  }
  if (qm->QMmethod==eQMmethodCASSCF){
    /* fetch guess from checkpoint file, always for CASSCF */
    fprintf(out,"%s"," guess=read");
  }
  fprintf(out,"\nNosymm units=bohr\n");
  
  if(qm->bTS){
    fprintf(out,"OPT=(Redundant,TS,noeigentest,ModRedundant) Punch=(Coord,Derivatives) ");
  }
  else if (qm->bOPT){
    fprintf(out,"OPT=(Redundant,ModRedundant) Punch=(Coord,Derivatives) ");
  }
  else{
    fprintf(out,"FORCE Punch=(Derivatives) ");
  }
  fprintf(out,"iop(3/33=1)\n\n");
  fprintf(out, "input-file generated by gromacs\n\n");
  fprintf(out,"%2d%2d\n",qm->QMcharge,qm->multiplicity);
  for (i=0;i<qm->nrQMatoms;i++){
#ifdef GMX_DOUBLE
    fprintf(out,"%3d %10.7lf  %10.7lf  %10.7lf\n",
	    qm->atomicnumberQM[i],
	    qm->xQM[i][XX]/BOHR2NM,
	    qm->xQM[i][YY]/BOHR2NM,
	    qm->xQM[i][ZZ]/BOHR2NM);
#else
    fprintf(out,"%3d %10.7f  %10.7f  %10.7f\n",
	    qm->atomicnumberQM[i],
	    qm->xQM[i][XX]/BOHR2NM,
	    qm->xQM[i][YY]/BOHR2NM,
	    qm->xQM[i][ZZ]/BOHR2NM);
#endif
  }

  /* Pseudo Potential and ECP are included here if selected (MEthod suffix LAN) */
  if(qm->QMmethod==eQMmethodB3LYPLAN){
    fprintf(out,"\n");
    for(i=0;i<qm->nrQMatoms;i++){
      if(qm->atomicnumberQM[i]<21){
	fprintf(out,"%d ",i+1);
      }
    }
    fprintf(out,"\n%s\n****\n",eQMbasis_names[qm->QMbasis]);
    
    for(i=0;i<qm->nrQMatoms;i++){
      if(qm->atomicnumberQM[i]>21){
	fprintf(out,"%d ",i+1);
      }
    }
    fprintf(out,"\n%s\n****\n\n","lanl2dz");    
    
    for(i=0;i<qm->nrQMatoms;i++){
      if(qm->atomicnumberQM[i]>21){
	fprintf(out,"%d ",i+1);
      }
    }
    fprintf(out,"\n%s\n","lanl2dz");    
  }    
  
  /* MM point charge data */
  if(QMMMrec->QMMMscheme!=eQMMMschemeoniom && mm->nrMMatoms){
//    fprintf(stderr,"nr mm atoms in gaussian.c = %d\n",mm->nrMMatoms);
    fprintf(out,"\n");
    if(qm->bTS||qm->bOPT){
      /* freeze the frontier QM atoms and Link atoms. This is
       * important only if a full QM subsystem optimization is done
       * with a frozen MM environmeent. For dynamics, or gromacs's own
       * optimization routines this is not important.
       */
      for(i=0;i<qm->nrQMatoms;i++){
	if(qm->frontatoms[i]){
	  fprintf(out,"%d F\n",i+1); /* counting from 1 */
	}
      }
      /* MM point charges include LJ parameters in case of QM optimization
       */
      for(i=0;i<mm->nrMMatoms;i++){
#ifdef GMX_DOUBLE
	fprintf(out,"%10.7lf  %10.7lf  %10.7lf %8.4lf 0.0 %10.7lf %10.7lf\n",
		mm->xMM[i][XX]/BOHR2NM,
		mm->xMM[i][YY]/BOHR2NM,
		mm->xMM[i][ZZ]/BOHR2NM,
		mm->MMcharges[i],
		mm->c6[i],mm->c12[i]);
#else
	fprintf(out,"%10.7f  %10.7f  %10.7f %8.4f 0.0 %10.7f %10.7f\n",
		mm->xMM[i][XX]/BOHR2NM,
		mm->xMM[i][YY]/BOHR2NM,
		mm->xMM[i][ZZ]/BOHR2NM,
		mm->MMcharges[i],
		mm->c6[i],mm->c12[i]);
#endif
      }
      fprintf(out,"\n");
    }
    else{
      for(i=0;i<mm->nrMMatoms;i++){
#ifdef GMX_DOUBLE
	fprintf(out,"%10.7lf  %10.7lf  %10.7lf %8.4lf\n",
		mm->xMM[i][XX]/BOHR2NM,
		mm->xMM[i][YY]/BOHR2NM,
		mm->xMM[i][ZZ]/BOHR2NM,
		mm->MMcharges[i]);
#else
	fprintf(out,"%10.7f  %10.7f  %10.7f %8.4f\n",
		mm->xMM[i][XX]/BOHR2NM,
		mm->xMM[i][YY]/BOHR2NM,
		mm->xMM[i][ZZ]/BOHR2NM,
		mm->MMcharges[i]);
#endif
      }
    }
  }
  fprintf(out,"\n");
  

  fclose(out);

}  /* write_gaussian_input_QED */

real read_gaussian_output_QED(t_commrec *cr,rvec QMgrad_S1[],rvec MMgrad_S1[],
			      rvec QMgrad_S0[],rvec MMgrad_S0[],int step,
			      t_QMrec *qm, t_MMrec *mm,rvec *tdm,
                              rvec tdmX[], rvec tdmY[], rvec tdmZ[],
                              rvec tdmXMM[], rvec tdmYMM[], rvec tdmZMM[],
                              real *Eground)
{
  int
    i,j,atnum;
  char
    buf[3000],*buf2;
  real
    QMener,rinv,qtdm,ri,ro,cosa;
  FILE
    *in_S1,*in_S0;

  if (MULTISIM(cr)){ 
    chdir (qm->subdir);
  } 
  in_S1=fopen("S1.7","r");

  /* the next line is the energy and in the case of CAS, the energy
   * difference between the two states.
   */
  if(NULL == fgets(buf,3000,in_S1)){
      gmx_fatal(FARGS,"Error reading Gaussian output");
  }

#ifdef GMX_DOUBLE
  sscanf(buf,"%lf\n",&QMener);
#else
  sscanf(buf,"%f\n", &QMener);
#endif
  /* next lines contain the excited state gradients of the QM atoms */
  for(i=0;i<qm->nrQMatoms;i++){
    if(NULL == fgets(buf,3000,in_S1)){
	gmx_fatal(FARGS,"Error reading Gaussian output");
    }
#ifdef GMX_DOUBLE
    sscanf(buf,"%lf %lf %lf\n",
	   &QMgrad_S1[i][XX],
	   &QMgrad_S1[i][YY],
	   &QMgrad_S1[i][ZZ]);
#else
    sscanf(buf,"%f %f %f\n",
	   &QMgrad_S1[i][XX],
	   &QMgrad_S1[i][YY],
	   &QMgrad_S1[i][ZZ]);
#endif     
  }
  /* the next lines are the gradients of the MM atoms */
  for(i=0;i<mm->nrMMatoms;i++){
    if(NULL==fgets(buf,3000,in_S1)){
      gmx_fatal(FARGS,"Error reading Gaussian output");
    }
#ifdef GMX_DOUBLE
    sscanf(buf,"%lf %lf %lf\n",
    &MMgrad_S1[i][XX],
    &MMgrad_S1[i][YY],
    &MMgrad_S1[i][ZZ]);
#else
    sscanf(buf,"%f %f %f\n",
    &MMgrad_S1[i][XX],
    &MMgrad_S1[i][YY],
    &MMgrad_S1[i][ZZ]);
#endif	
  }
  /* now comes the transition dipole moments  */
  if(NULL==fgets(buf,3000,in_S1)){
      gmx_fatal(FARGS,"Error reading Gaussian output");
    }
#ifdef GMX_DOUBLE
  sscanf(buf,"%lf %lf %lf\n",
	     &tdm[0][XX],
	     &tdm[0][YY],
	     &tdm[0][ZZ]);
#else
  sscanf(buf,"%f %f %f\n",
	 &tdm[0][XX],
         &tdm[0][YY],
	 &tdm[0][ZZ]);
#endif	

/* now we need to check if the dipole moment has changed sign. If so, we simply
 * also change the sign of the E field. We use the trick Dmitry used:
 */
  if(step){
    ri = sqrt(tdm[0][XX]*tdm[0][XX]+tdm[0][YY]*tdm[0][YY]+tdm[0][ZZ]*tdm[0][ZZ]);
    ro = sqrt(qm->tdmold[XX]*qm->tdmold[XX]+qm->tdmold[YY]*qm->tdmold[YY]+qm->tdmold[ZZ]*qm->tdmold[ZZ]);
    cosa = (tdm[0][XX]*qm->tdmold[XX]+tdm[0][YY]*qm->tdmold[YY]+tdm[0][ZZ]*qm->tdmold[ZZ]) / (ri * ro);
//    fprintf(stderr,"tdm = {%f,%f,%f}\n",tdm[0][XX],tdm[0][YY],tdm[0][ZZ]);
//    fprintf(stderr,"old = {%f,%f,%f}\n", qm->tdmold[XX], qm->tdmold[YY], qm->tdmold[ZZ]);
//    fprintf(stderr,"dotprod = %lf\n",cosa);
    if (cosa<0.0){
      fprintf(stderr, "Changing Efield sign\n");
      for (i=0;i<DIM;i++){
        qm->E[i]*=-1.0;
      }
    }
  }
  /* store the TDM in QMrec for the next step */
  qm->tdmold[XX] = tdm[0][XX];
  qm->tdmold[YY] = tdm[0][YY];
  qm->tdmold[ZZ] = tdm[0][ZZ];
  /* works only in combination with TeraChem
   */
  /* read in sequence nabla tdm[j]_ia
   * first X, then Y then Z
   */
  /* Read in the dipole moment gradients */

  for (i = 0 ; i< qm->nrQMatoms; i++ ){
    if(NULL==fgets(buf,3000,in_S1)){
      gmx_fatal(FARGS,"Error reading Gaussian output");
    }
#ifdef GMX_DOUBLE
    sscanf(buf,"%lf %lf %lf\n",&tdmX[i][0],&tdmX[i][1],&tdmX[i][2]);
#else
    sscanf(buf,"%f %f %f\n",&tdmX[i][0],&tdmX[i][1],&tdmX[i][2]);
#endif
  }
  for (i = 0 ; i< qm->nrQMatoms; i++ ){
    if(NULL==fgets(buf,3000,in_S1)){
      gmx_fatal(FARGS,"Error reading Gaussian output");
    }
#ifdef GMX_DOUBLE
    sscanf(buf,"%lf %lf %lf\n",&tdmY[i][0],&tdmY[i][1],&tdmY[i][2]);
#else
    sscanf(buf,"%f %f %f\n",&tdmY[i][0],&tdmY[i][1],&tdmY[i][2]);
#endif
  }
  for (i = 0 ; i< qm->nrQMatoms; i++ ){
    if(NULL==fgets(buf,3000,in_S1)){
      gmx_fatal(FARGS,"Error reading TDMZ from Gaussian output");
    }
#ifdef GMX_DOUBLE
    sscanf(buf,"%lf %lf %lf\n",&tdmZ[i][0],&tdmZ[i][1],&tdmZ[i][2]);
#else
    sscanf(buf,"%f %f %f\n",&tdmZ[i][0],&tdmZ[i][1],&tdmZ[i][2]);
#endif
  }

/* now comes the MM TDM gradients */
  for (i = 0 ; i< mm->nrMMatoms; i++ ){
    if(NULL==fgets(buf,3000,in_S1)){
      gmx_fatal(FARGS,"Error reading MM TDM grad from  Gaussian output");
    }
#ifdef GMX_DOUBLE
    sscanf(buf,"%lf %lf %lf\n",&tdmXMM[i][0],&tdmXMM[i][1],&tdmXMM[i][2]);
#else
    sscanf(buf,"%f %f %f\n",&tdmXMM[i][0],&tdmXMM[i][1],&tdmXMM[i][2]);
#endif
  }
  for (i = 0 ; i< mm->nrMMatoms; i++ ){
    if(NULL==fgets(buf,3000,in_S1)){
      gmx_fatal(FARGS,"Error reading Gaussian output");
    }
#ifdef GMX_DOUBLE
    sscanf(buf,"%lf %lf %lf\n",&tdmYMM[i][0],&tdmYMM[i][1],&tdmYMM[i][2]);
#else
    sscanf(buf,"%f %f %f\n",&tdmYMM[i][0],&tdmYMM[i][1],&tdmYMM[i][2]);
#endif
  }
  for (i = 0 ; i< mm->nrMMatoms; i++ ){
    if(NULL==fgets(buf,3000,in_S1)){
      gmx_fatal(FARGS,"Error reading TDMZ from Gaussian output");
    }
#ifdef GMX_DOUBLE
    sscanf(buf,"%lf %lf %lf\n",&tdmZMM[i][0],&tdmZMM[i][1],&tdmZMM[i][2]);
#else
    sscanf(buf,"%f %f %f\n",&tdmZMM[i][0],&tdmZMM[i][1],&tdmZMM[i][2]);
#endif
  }

  in_S0=fopen("S0.7","r");
  if (in_S0==NULL)
    gmx_fatal(FARGS,"Error reading Gaussian output");
  /* now read in ground state information from a second file */
  if(NULL == fgets(buf,3000,in_S0)){
      gmx_fatal(FARGS,"Error reading Gaussian output");
  }
  
#ifdef GMX_DOUBLE
  sscanf(buf,"%lf\n",Eground);
#else
  sscanf(buf,"%f\n", Eground);
#endif
  /* next lines contain the excited state gradients of the QM atoms */
  for(i=0;i<qm->nrQMatoms;i++){
    if(NULL == fgets(buf,3000,in_S0)){
	gmx_fatal(FARGS,"Error reading Gaussian output");
    }
#ifdef GMX_DOUBLE
    sscanf(buf,"%lf %lf %lf\n",
	   &QMgrad_S0[i][XX],
	   &QMgrad_S0[i][YY],
	   &QMgrad_S0[i][ZZ]);
#else
    sscanf(buf,"%f %f %f\n",
	   &QMgrad_S0[i][XX],
	   &QMgrad_S0[i][YY],
	   &QMgrad_S0[i][ZZ]);
#endif     
  }
  /* the next lines are the gradients of the MM atoms */
  for(i=0;i<mm->nrMMatoms;i++){
    if(NULL==fgets(buf,3000,in_S0)){
	gmx_fatal(FARGS,"Error reading Gaussian output");
    }
#ifdef GMX_DOUBLE
      sscanf(buf,"%lf %lf %lf\n",
	     &MMgrad_S0[i][XX],
	     &MMgrad_S0[i][YY],
	     &MMgrad_S0[i][ZZ]);
#else
      sscanf(buf,"%f %f %f\n",
	     &MMgrad_S0[i][XX],
	     &MMgrad_S0[i][YY],
	     &MMgrad_S0[i][ZZ]);
#endif	
  }
  fclose(in_S0);
  fclose(in_S1);
  return(QMener);  
} /* read_gaussian_output_QED */

real read_gaussian_output(rvec QMgrad[],rvec MMgrad[],int step,
			  t_QMrec *qm, t_MMrec *mm)
{
  int
    i,j,atnum;
  char
    buf[300];
  real
    QMener;
  FILE
    *in;
  
  in=fopen("fort.7","r");

  /* in case of an optimization, the coordinates are printed in the
   * fort.7 file first, followed by the energy, coordinates and (if
   * required) the CI eigenvectors.
   */
  if(qm->bTS||qm->bOPT){
    for(i=0;i<qm->nrQMatoms;i++){
      if( NULL == fgets(buf,300,in)){
	  gmx_fatal(FARGS,"Error reading Gaussian output - not enough atom lines?");
      }

#ifdef GMX_DOUBLE
      sscanf(buf,"%d %lf %lf %lf\n",
	     &atnum,
	     &qm->xQM[i][XX],
	     &qm->xQM[i][YY],
	     &qm->xQM[i][ZZ]);
#else
      sscanf(buf,"%d %f %f %f\n",
	     &atnum,
	     &qm->xQM[i][XX],
	     &qm->xQM[i][YY],
	     &qm->xQM[i][ZZ]);
#endif     
      for(j=0;j<DIM;j++){
	qm->xQM[i][j]*=BOHR2NM;
      }
    }
  }
  /* the next line is the energy and in the case of CAS, the energy
   * difference between the two states.
   */
  if(NULL == fgets(buf,300,in)){
      gmx_fatal(FARGS,"Error reading Gaussian output");
  }

#ifdef GMX_DOUBLE
  sscanf(buf,"%lf\n",&QMener);
#else
  sscanf(buf,"%f\n", &QMener);
#endif
  /* next lines contain the gradients of the QM atoms */
  for(i=0;i<qm->nrQMatoms;i++){
    if(NULL == fgets(buf,300,in)){
	gmx_fatal(FARGS,"Error reading Gaussian output");
    }
#ifdef GMX_DOUBLE
    sscanf(buf,"%lf %lf %lf\n",
	   &QMgrad[i][XX],
	   &QMgrad[i][YY],
	   &QMgrad[i][ZZ]);
#else
    sscanf(buf,"%f %f %f\n",
	   &QMgrad[i][XX],
	   &QMgrad[i][YY],
	   &QMgrad[i][ZZ]);
#endif     
  }
  /* the next lines are the gradients of the MM atoms */
  if(qm->QMmethod>=eQMmethodRHF){  
    for(i=0;i<mm->nrMMatoms;i++){
      if(NULL==fgets(buf,300,in)){
          gmx_fatal(FARGS,"Error reading Gaussian output");
      }
#ifdef GMX_DOUBLE
      sscanf(buf,"%lf %lf %lf\n",
	     &MMgrad[i][XX],
	     &MMgrad[i][YY],
	     &MMgrad[i][ZZ]);
#else
      sscanf(buf,"%f %f %f\n",
	     &MMgrad[i][XX],
	     &MMgrad[i][YY],
	     &MMgrad[i][ZZ]);
#endif	
    }
  }
  fclose(in);
  return(QMener);  
}

real read_gaussian_SH_output(rvec QMgrad[],rvec MMgrad[],int step,
			     gmx_bool swapped,t_QMrec *qm, t_MMrec *mm,real *DeltaE)
{
  int
    i;
  char
    buf[300];
  real
    QMener;
  FILE
    *in;
  
  in=fopen("fort.7","r");
  /* first line is the energy and in the case of CAS, the energy
   * difference between the two states.
   */
  if(NULL == fgets(buf,300,in)){
      gmx_fatal(FARGS,"Error reading Gaussian output");
  }

#ifdef GMX_DOUBLE
  sscanf(buf,"%lf %lf\n",&QMener,DeltaE);
#else
  sscanf(buf,"%f %f\n",  &QMener,DeltaE);
#endif
  
  /* switch on/off the State Averaging */
  
  if(*DeltaE > qm->SAoff){
    if (qm->SAstep > 0){
      qm->SAstep--;
    }
  }
  else if (*DeltaE < qm->SAon || (qm->SAstep > 0)){
    if (qm->SAstep < qm->SAsteps){
      qm->SAstep++;
    }
  }
  
  /* for debugging: */
  fprintf(stderr,"Gap = %5f,SA = %3d\n",*DeltaE,(qm->SAstep>0));
  /* next lines contain the gradients of the QM atoms */
  for(i=0;i<qm->nrQMatoms;i++){
    if(NULL==fgets(buf,300,in)){
	gmx_fatal(FARGS,"Error reading Gaussian output");
    }

#ifdef GMX_DOUBLE
    sscanf(buf,"%lf %lf %lf\n",
	   &QMgrad[i][XX],
	   &QMgrad[i][YY],
	   &QMgrad[i][ZZ]);
#else
    sscanf(buf,"%f %f %f\n",
	   &QMgrad[i][XX],
	   &QMgrad[i][YY],
	   &QMgrad[i][ZZ]);
#endif     
  }
  /* the next lines, are the gradients of the MM atoms */
  
  for(i=0;i<mm->nrMMatoms;i++){
    if(NULL==fgets(buf,300,in)){
	gmx_fatal(FARGS,"Error reading Gaussian output");
    }
#ifdef GMX_DOUBLE
    sscanf(buf,"%lf %lf %lf\n",
	   &MMgrad[i][XX],
	   &MMgrad[i][YY],
	   &MMgrad[i][ZZ]);
#else
    sscanf(buf,"%f %f %f\n",
	   &MMgrad[i][XX],
	   &MMgrad[i][YY],
	   &MMgrad[i][ZZ]);
#endif	
  }
  
  /* the next line contains the two CI eigenvector elements */
  if(NULL==fgets(buf,300,in)){
      gmx_fatal(FARGS,"Error reading Gaussian output");
  }
  if(!step){
    sscanf(buf,"%d",&qm->CIdim);
    snew(qm->CIvec1,qm->CIdim);
    snew(qm->CIvec1old,qm->CIdim);
    snew(qm->CIvec2,qm->CIdim);
    snew(qm->CIvec2old,qm->CIdim);
  } else {
    /* before reading in the new current CI vectors, copy the current
     * CI vector into the old one.
     */
    for(i=0;i<qm->CIdim;i++){
      qm->CIvec1old[i] = qm->CIvec1[i];
      qm->CIvec2old[i] = qm->CIvec2[i];
    }
  }
  /* first vector */
  for(i=0;i<qm->CIdim;i++){
    if(NULL==fgets(buf,300,in)){
	gmx_fatal(FARGS,"Error reading Gaussian output");
    }
#ifdef GMX_DOUBLE
    sscanf(buf,"%lf\n",&qm->CIvec1[i]);
#else
    sscanf(buf,"%f\n", &qm->CIvec1[i]);   
#endif
  }
  /* second vector */
  for(i=0;i<qm->CIdim;i++){
    if(NULL==fgets(buf,300,in)){
	gmx_fatal(FARGS,"Error reading Gaussian output");
    }
#ifdef GMX_DOUBLE
    sscanf(buf,"%lf\n",&qm->CIvec2[i]);
#else
    sscanf(buf,"%f\n", &qm->CIvec2[i]);   
#endif
  }
  fclose(in);
  return(QMener);  
}

real inproduct(real *a, real *b, int n){
  int
    i;
  real
    dot=0.0;
  
  /* computes the inner product between two vectors (a.b), both of
   * which have length n.
   */  
  for(i=0;i<n;i++){
    dot+=a[i]*b[i];
  }
  return(dot);
}

dplx inproduct_complex(dplx *a, dplx *b, int n){
  int
    i;
  dplx
    dot=0.0+IMAG*0.0;
  
  for(i=0;i<n;i++){
    dot+=conj(a[i])*b[i];
  }
  return(dot);
}

int hop(int step, t_QMrec *qm)
{
  int
    swap = 0;
  real
    d11=0.0,d12=0.0,d21=0.0,d22=0.0;
  
  /* calculates the inproduct between the current Ci vector and the
   * previous CI vector. A diabatic hop will be made if d12 and d21
   * are much bigger than d11 and d22. In that case hop returns true,
   * otherwise it returns false.
   */  
  if(step){ /* only go on if more than one step has been done */
    d11 = inproduct(qm->CIvec1,qm->CIvec1old,qm->CIdim);
    d12 = inproduct(qm->CIvec1,qm->CIvec2old,qm->CIdim);
    d21 = inproduct(qm->CIvec2,qm->CIvec1old,qm->CIdim);
    d22 = inproduct(qm->CIvec2,qm->CIvec2old,qm->CIdim);
  }
  fprintf(stderr,"-------------------\n");
  fprintf(stderr,"d11 = %13.8f\n",d11);
  fprintf(stderr,"d12 = %13.8f\n",d12);
  fprintf(stderr,"d21 = %13.8f\n",d21);
  fprintf(stderr,"d22 = %13.8f\n",d22);
  fprintf(stderr,"-------------------\n");
  
  if((fabs(d12)>0.2)&&(fabs(d21)>0.2))
    swap = 1;
  
  return(swap);
}

void do_gaussian(int step,char *exe)
{
  char
    buf[300];

  /* make the call to the gaussian binary through system()
   * The location of the binary will be picked up from the 
   * environment using getenv().
   */
  if(step) /* hack to prevent long inputfiles */
    sprintf(buf,"%s < %s > %s",
	    exe,
	    "input.com",
	    "input.log");
  else
    sprintf(buf,"%s < %s > %s",
	    exe,
            "input.com",
	    "input.log");
//  fprintf(stderr,"Calling '%s'\n",buf);
#ifdef GMX_NO_SYSTEM
  printf("Warning-- No calls to system(3) supported on this platform.");
  gmx_fatal(FARGS,"Call to '%s' failed\n",buf);
#else
  if ( system(buf) != 0 )
    gmx_fatal(FARGS,"Call to '%s' failed\n",buf);
#endif
}

real call_gaussian(t_commrec *cr,  t_forcerec *fr, 
		   t_QMrec *qm, t_MMrec *mm, rvec f[], rvec fshift[])
{
  /* normal gaussian jobs */
  static int
    step=0;
  int
    i,j;
  real
    QMener=0.0;
  rvec
    *QMgrad,*MMgrad;
  char
    *exe;
  
  snew(exe,30);
  sprintf(exe,"%s/%s",qm->gauss_dir,qm->gauss_exe);
  snew(QMgrad,qm->nrQMatoms);
  snew(MMgrad,mm->nrMMatoms);

  write_gaussian_input(step,fr,qm,mm);
  do_gaussian(step,exe);
  QMener = read_gaussian_output(QMgrad,MMgrad,step,qm,mm);
  /* put the QMMM forces in the force array and to the fshift
   */
  for(i=0;i<qm->nrQMatoms;i++){
    for(j=0;j<DIM;j++){
      f[i][j]      = HARTREE_BOHR2MD*QMgrad[i][j];
      fshift[i][j] = HARTREE_BOHR2MD*QMgrad[i][j];
    }
  }
  for(i=0;i<mm->nrMMatoms;i++){
    for(j=0;j<DIM;j++){
      f[i+qm->nrQMatoms][j]      = HARTREE_BOHR2MD*MMgrad[i][j];      
      fshift[i+qm->nrQMatoms][j] = HARTREE_BOHR2MD*MMgrad[i][j];
    }
  }
  QMener = QMener*HARTREE2KJ*AVOGADRO;
  step++;
  free(exe);
  free(QMgrad);
  free(MMgrad);
  return(QMener);
} /* call_gaussian */

typedef struct {
  int j;
  int i;
} t_perm;

void track_states(dplx *vecold, dplx *vecnew, int ndim){
  fprintf(stderr,"Call to track_states\n");
  int
    *stmap,i,j,k;
  double
    maxover,ri,ro,sina,cosa;

  snew(stmap,ndim);
  for(i=0;i<ndim;i++){
    maxover=-1.0;
    for(j=0;j<ndim;j++){
///      if (fabs(inproduct(&vecnew[i*ndim], &vecold[j*ndim], ndim))>maxover) {
///        maxover=fabs(inproduct(&vecnew[i*ndim], &vecold[j*ndim], ndim));
      if (cabs(inproduct_complex(&vecnew[i*ndim], &vecold[j*ndim], ndim))>maxover) {
	maxover=cabs(inproduct_complex(&vecnew[i*ndim], &vecold[j*ndim], ndim));
	stmap[i]=j;
//	fprintf(stderr,"From track_states: maxover_%d=%lf\n",j,maxover);
      }
    }
  }
  for(i=0;i<ndim;i++){
///    ri = sqrt(inproduct(&vecnew[i*ndim], &vecnew[i*ndim],ndim));
///    ro = sqrt(inproduct(&vecold[stmap[i]*ndim], &vecold[stmap[i]*ndim],ndim));
///    cosa = inproduct(&vecnew[i*ndim], &vecold[stmap[i]*ndim],ndim)/ (ri * ro);
    ri = sqrt(inproduct_complex(&vecnew[i*ndim], &vecnew[i*ndim],ndim));
    ro = sqrt(inproduct_complex(&vecold[stmap[i]*ndim], &vecold[stmap[i]*ndim],ndim));
    cosa = creal(inproduct_complex(&vecnew[i*ndim], &vecold[stmap[i]*ndim],ndim))/(ri*ro);
//    fprintf(stderr,"cosa=%lf\n",cosa);
    if (cosa<0.0){
      for(j=0;j<ndim;j++){
	vecnew[i*ndim+j]=-vecnew[i*ndim+j];
      }
    }
  }
  fprintf(stderr,"Done with track_states\n");
  sfree(stmap);
}
   
int trace_states(t_QMrec *qm, dplx *c, double *eigvec, int ndim){
  int
    i,j,k,current,duplicate,*states,permute,hopto,nperm=0,npermnew=0;
////  double
////    overlap=0.0,*tempeigvec=NULL;
  dplx
    overlap=0.0,*tempeigvec=NULL;
  t_perm 
    *perm=NULL,*permnew=NULL;
  dplx
    *cdummy;

  current=qm->polariton;
  hopto=current;
  snew(cdummy,ndim);
  for (i=0;i<ndim;i++){
    cdummy[i]=c[i];
  }
  snew(perm,ndim*ndim);
  /* check if the active state passed through an unavoided trivial crossing 
   */
  for(i=0;i<ndim;i++){
    overlap=0.0;
    for(k=0;k<ndim;k++){
        overlap+=eigvec[i*ndim+k]*(qm->eigvec[current*ndim+k]);
    }
    /* 0.9 overlap is abitrariy, needs some testing...
     */
////    if((fabs(overlap) > 0.9) && (i != current)){
    if((abs(overlap) > 0.9) && (i != current)){
      fprintf(stderr,"trivial hopping, enforce diabatic hop by flipping %d and %d\n",current,i);
        hopto=i;
    }
  }
  /* check if also the non-active states passed through trival crossings
   */
  for(i=0;i<ndim;i++){
    for(j=0;j<ndim;j++){
      overlap=0.0;
      for(k=0;k<ndim;k++){
        overlap+=eigvec[i*ndim+k]*(qm->eigvec[j*ndim+k]);
      }
////      if((fabs(overlap) > 0.9) && (i != j)){
      if((abs(overlap) > 0.9) && (i != j)){
        fprintf (stderr,"Trivial crossing between states %d and %d. Switching coefficients \n",i,j);
        perm[nperm].i=i;
        perm[nperm].j=j;
        nperm++;
      }
    }
  }
  /* correct the (arbitrary) signs of the eigenvectors
   */ 
  track_states(qm->eigvec, eigvec, ndim);
  /* With the information on the unavoided trivial crossings, 
   * re-order the time-dependent coefficients (and populations)
   * based on the new order of states. THis code could probably 
   * be written more compact or smarter, but seems to do the trick.
   */
  snew(permnew,nperm);
  for (i=0;i<nperm;i++){
    duplicate=0;
    for (j=i+1;j<nperm;j++){
      if (perm[i].i==perm[j].j && perm[i].j==perm[j].i){
        duplicate=1;
      }
    }
    if (!duplicate){
      permnew[npermnew].i=perm[i].i;
      permnew[npermnew].j=perm[i].j;
      npermnew++;
    }
  }
  snew(tempeigvec,ndim*ndim);
  for (i=0;i<ndim*ndim;i++){
    tempeigvec[i]=qm->eigvec[i];
  }
  snew(states,ndim);
  for ( i = 0 ; i < ndim ; i++ ){
    states[i]  = i;
  }
  for(i=0;i<npermnew;i++){
    k = states[permnew[i].i];
    states[permnew[i].i]= states[permnew[i].j];
    states[permnew[i].j]= k;
  }
  /* then, renumber based on i and j. 
   * motivation is that unless number of flips is even
   * this would be sufficient, but sometimes it is not.
   */
  for(i=0;i<npermnew;i++){
    states[permnew[i].i]= permnew[i].j;
  }
  fprintf(stderr,"states: ");
  for ( i = 0 ; i < ndim ; i++ ){
    fprintf(stderr,"%d ",states[i]);
    c[i]=cdummy[states[i]];
    for(k=0;k<ndim;k++){
      qm->eigvec[i*ndim+k] = tempeigvec[states[i]*ndim+k];
    }
  }
  fprintf(stderr,"\n");
//  fprintf(stderr,"C: ");
//  for(i=0;i<ndim;i++){
//    fprintf (stderr,"%lf ",conj(c[i])*c[i]);
//  }
//  fprintf(stderr,"\n ");
  free (tempeigvec);
  free (permnew);
  free(states);
  free(perm);
  free(cdummy);
  return(hopto); 
}  


int QEDFSSHop(int step, t_QMrec *qm, dplx *eigvec, int ndim, double *eigval, real dt,t_QMMMrec *qr){
  fprintf(stderr,"Call to QEDFSSHop\n");
  int
    i,j,k,current,hopto;
  double
    *f,*p,b,rnr,ptot=0.0,invdt,overlap=0.0,btot=0.0;
  dplx 
     *g=NULL,*c=NULL,*cold=NULL,*U=NULL; 
//  FILE
//    *Cout=NULL;
//   char
//     buf[5000];

  invdt=1.0/dt*AU2PS;
  current = qm->polariton;
  hopto=qm->polariton;
  snew(c,ndim); 
  snew(cold,ndim);
  if(step){
    rnr = qm->rnr[step];
    for (i=0;i<ndim;i++){
       cold[i]=qm->creal[i]+IMAG*qm->cimag[i];
       c[i]=qm->creal[i]+IMAG*qm->cimag[i];
//       fprintf(stderr,"|c[%d]|^2 = %lf ;",i,conj(c[i])*c[i]);
//       fprintf(stderr,"|g[%d]|^2 = %lf\n",i,conj(cold[i])*cold[i]);
    }
    snew(p,ndim);
    
    /* check for trivial hops and trace the states 
     */
    track_states(qm->eigvec,eigvec,ndim);
//    hopto = trace_states(qm,c,eigvec,ndim);
    if (hopto != current){
      /* we thus have a diabatic hop, and enforce this hop.
       * We still propagae the wavefunction
       */
      rnr=1000;
    }
    /* make choice between hopping in adiabatic or diabatic basis i
     * will be added as an option to the mdp file
     */
    if (qr->SHmethod==eSHmethodGranucci) {
    /* implementation of Tully's FFSH in adiabatic basis
     * following Granucci, Persico and Toniolo
     * J. CHem. Phys. 114 (2001) 10608  
     */
      //track_states(qm->eigvec, eigvec, ndim);

      snew(U,ndim*ndim);
      /* we need to keep the coefficients at t, as we need both c(t) and
       * c(t+dt) to compute the hopping probanilities. We just make a copy
       */
      propagate_local_dia(ndim,dt,c,eigvec,qm->eigvec,eigval,qm->eigval,U);
      fprintf(stderr," population that leaves state %d: %lf\n",current,(conj(cold[current])*cold[current]-conj(c[current])*c[current]));
      fprintf(stderr, "probability to leave state %d is %lf\n",current,(conj(cold[current])*cold[current]-conj(c[current])*c[current])/(conj(cold[current])*cold[current]));
      ptot=(conj(cold[current])*cold[current]-conj(c[current])*c[current])/(conj(cold[current])*cold[current]);
      if (ptot<=0){
        for ( i = 0 ; i < ndim ; i++ ){
          p[i] = 0;
        }
      } 
      else{
        btot=0.0;
        for ( i = 0 ; i < ndim ; i++ ){
          if ( i != current ){
            b = conj(U[i*ndim+current]*cold[current])*U[i*ndim+current]*cold[current];
            if (b>0.0){
              btot+=b;
              p[i]=b;
            }
            fprintf(stderr,"from state %d to state %d, b = %lf\n",current,i,b);
          }
        }
        for (i = 0 ;i<ndim;i++){
          p[i]=p[i]/btot*ptot;
        }
      }
      free(U);
    }
    else{
      /* implementation of Tully's FSSH in adiabatic basis 
       * Following Fabiano, Keal and Thiel
       * Chem. Phys. 2008 
       */ 
      snew(f,ndim*ndim);
      for(i=0;i<ndim;i++){
	for(j=i+1;j<ndim;j++){
	  for (k=0;k<ndim;k++){
///	    f[i*ndim+j]+=0.5*invdt*(qm->eigvec[i*ndim+k]*eigvec[j*ndim+k]-eigvec[i*ndim+k]*qm->eigvec[j*ndim+k]);
///	    f[j*ndim+i]+=0.5*invdt*(qm->eigvec[j*ndim+k]*eigvec[i*ndim+k]-eigvec[j*ndim+k]*qm->eigvec[i*ndim+k]);
	    f[i*ndim+j]+=0.5*invdt*(conj(qm->eigvec[i*ndim+k])*eigvec[j*ndim+k]-conj(eigvec[i*ndim+k])*qm->eigvec[j*ndim+k]);
	    f[j*ndim+i]+=0.5*invdt*(conj(qm->eigvec[j*ndim+k])*eigvec[i*ndim+k]-conj(eigvec[j*ndim+k])*qm->eigvec[i*ndim+k]);
	    fprintf(stderr,"From Tully's FSSH in QEDFSSHop, copy_of_f[%d,%d]=%lf+%lfI\n",i,j,creal(0.5*invdt*(conj(qm->eigvec[i*ndim+k])*eigvec[j*ndim+k]-conj(eigvec[i*ndim+k])*qm->eigvec[j*ndim+k])),cimag(0.5*invdt*(conj(qm->eigvec[i*ndim+k])*eigvec[j*ndim+k]-conj(eigvec[i*ndim+k])*qm->eigvec[j*ndim+k])));
	    fprintf(stderr,"From Tully's FSSH in QEDFSSHop, copy_of_f[%d,%d]=%lf+%lfI\n",j,i,creal(0.5*invdt*(conj(qm->eigvec[j*ndim+k])*eigvec[i*ndim+k]-conj(eigvec[j*ndim+k])*qm->eigvec[i*ndim+k])),cimag(0.5*invdt*(conj(qm->eigvec[j*ndim+k])*eigvec[i*ndim+k]-conj(eigvec[j*ndim+k])*qm->eigvec[i*ndim+k])));
	  }  
	}
      }
      propagate(ndim,dt, c,eigvec,qm->eigvec,eigval,qm->eigval);
      /* following Tully, Thiel et al. seem to do the propagation after 
       * the computation of hoping 
       */ 
      for(i=0;i<ndim;i++){
        if(i != current ){
          b=2*creal(conj(c[current])*c[i]*f[current*ndim+i]);
          if(b>0){
            p[i]=b*dt/AU2PS/(conj(c[current])*c[current]);
          }
        }
      }
      free(f);
    }
    if (rnr < 1000){/* avoid double hopping in case of trivial diabatic hops */
      ptot=0.0;
      for(i=0;i<ndim;i++){
        if ( i != current && ptot < rnr ){
          fprintf(stderr,"probability to hop from %d to %d is %lf\n",current,i,p[i]);
          if ( ptot+p[i] > rnr ) {
            hopto = i;
            fprintf(stderr,"hopping at step %d with probability %lf\n",step,ptot+p[i]);
          }
          ptot+=p[i];
        }
      }
    }
    /* store the expansion coefficients for the next step
     */
    for ( i = 0 ; i < ndim ; i++ ){
      qm->creal[i] = creal(c[i]);
      qm->cimag[i] = cimag(c[i]);
    } 
    /* some writinig
     */
    fprintf(stderr,"step %d: C: ",step);
//    sprintf(buf,"%s/C.dat",qm->work_dir);
//    Cout=fopen (buf,"w");
//    fprintf(Cout,"%d\n",step);
    for(i=0;i<ndim;i++){
      fprintf (stderr," %.5lf ",conj(c[i])*c[i]);    
//      fprintf (Cout,"%.5lf %.5lf\n ",qm->creal[i],qm->cimag[i]);
    }
//    fclose(Cout);
    fprintf(stderr,"\n");
    free(p);
  }
  else{
//    qm->creal[current]=1.0;
    if(qm->QEDrestart){
      track_states(qm->eigvec,eigvec,ndim);
    }
    fprintf(stderr,"step %d: C: ",step);
    for(i=0;i<ndim;i++){
      c[i] = qm->creal[i]+ IMAG*qm->cimag[i];
      fprintf (stderr,"%.5lf ",conj(c[i])*c[i]);
    }    
    fprintf(stderr,"\n");
  }
  free(c);
  free(cold);
  fprintf(stderr,"Done with QEDFSSHop\n");
  return(hopto);
}

int QEDhop(int step, t_QMrec *qm, dplx *eigvec, int ndim, double *eigval){
  fprintf(stderr,"call to QEDhop\n");
///  double
///    dii,dij,dij_max=0.0;
  dplx
    dii,dij,dij_max=0.0+IMAG*0.0;
  int
    i,k,current,hopto=0;

  /* check overlap with all states within the energy treshold that is
 *    * set by qm->SAon
 *       */
  hopto=qm->polariton;
  current=qm->polariton;
  for (i=0; i < ndim; i++){
    /* check if the energy gap is within the treshold */
    if(fabs(eigval[current]-eigval[i]) < qm->SAon){
///      dii  = 0.0;
///      dij  = 0.0;
      dii  = 0.0+IMAG*0.0;
      dij  = 0.0+IMAG*0.0;
      for (k=0;k<ndim;k++){
///	dii+=eigvec[current*ndim+k]*(qm->eigvec[current*ndim+k]);
///	dij+=eigvec[i*ndim+k]*(qm->eigvec[current*ndim+k]);
	dii+=conj(eigvec[current*ndim+k])*(qm->eigvec[current*ndim+k]);
	dij+=conj(eigvec[i*ndim+k])*(qm->eigvec[current*ndim+k]);
      }
///      if (fabs(dij) > fabs(dii)){
///        if(fabs(dij) > fabs(dij_max)){
      if (cabs(dij) > cabs(dii)){
	if(cabs(dij) > cabs(dij_max)){
	  hopto=i;
	  dij_max = dij;
	}
      }
      fprintf(stderr,"Overlap between %d and %d\n",current,i);
      fprintf(stderr,"-------------------\n");
///      fprintf(stderr,"dij = %13.8f\n",dij);
      fprintf(stderr,"dij = %13.8f+%13.8fI\n",creal(dij),cimag(dij));
      fprintf(stderr,"-------------------\n");
    }
  }
  if (current != hopto ){
/*    qm->polariton = hopto;*/
    fprintf (stderr,"hopping from state %d to state %d\n",current,hopto);
  }
  /* copy the current vectors to the old vectors!
 *    */
  for(i=0;i<ndim*ndim;i++){
    qm->eigvec[i]=eigvec[i];
  }
  fprintf(stderr,"QEDhop done\n");
  return (hopto);
}


void get_dipole_gradients(t_QMrec *qm, rvec E,rvec tdmgrad[],rvec tdm)
{
  real
    tm,mm,*mass=NULL;
  real 
    *w_rls=NULL;
  int
    i,j,k,l;
  rvec
    *xcopy,xcm,xrefcm,Erot,*tdmgrad_rot=NULL,*xfit;
  static int step=0;
  matrix
    R,Rinv,box;
  t_atoms
    *atoms=NULL;
  FILE
   *in;
  char 
    *buf,*tdmfile,*reffile,*reftitle;
  /* did not want to link atoms struct all way through, so do it luke this 
   * instead
   */
  if(!step){
    snew(buf,4000);
    snew(tdmfile,4000);
    snew(reffile,4000);
    snew(reftitle,4000);
    buf=getenv("TDM_FILE");
    if(buf)
      sscanf(buf,"%s",tdmfile);
    else
      gmx_fatal(FARGS,"no $TDM_FILE\nthis file (including path) contains the natoms TDM gradients in xyz format\nThe last line is the reference TDM\n");
     
    buf=getenv("REF_FILE");
    if(buf)
      sscanf(buf,"%s",reffile);
    else
      gmx_fatal(FARGS,"no $REF_FILE\nthis gro file (including path) contains the reference structure for which the TDM is known\n");

    snew(qm->xref,qm->nrQMatoms);
    snew(atoms,1);
    init_t_atoms(atoms,qm->nrQMatoms,0);
    read_stx_conf(reffile,reftitle,atoms,qm->xref,NULL,NULL,box);
    fprintf(stderr,"picking up TDM gradients (au) from file %s\n",tdmfile);
    /* last line contains the reference dipole moment */
    snew(qm->tdmXgrad,qm->nrQMatoms);
    snew(qm->tdmYgrad,qm->nrQMatoms);
    snew(qm->tdmZgrad,qm->nrQMatoms);

    in=fopen(tdmfile,"r");
    for(i=0;i<qm->nrQMatoms;i++){
      if(NULL == fgets(buf,3000,in)){
        gmx_fatal(FARGS,"Error reading TDM gradient file");
      }
#ifdef GMX_DOUBLE
      sscanf(buf,"%lf %lf %lf %lf %lf %lf %lf %lf %lf\n",
           &qm->tdmXgrad[i][XX],
           &qm->tdmXgrad[i][YY],
           &qm->tdmXgrad[i][ZZ],
           &qm->tdmYgrad[i][XX],
           &qm->tdmYgrad[i][YY],
           &qm->tdmYgrad[i][ZZ],
           &qm->tdmZgrad[i][XX],
           &qm->tdmZgrad[i][YY],
           &qm->tdmZgrad[i][ZZ]);
#else
      sscanf(buf,"%f %f %f %f %f %f %f %f %f\n",
           &qm->tdmXgrad[i][XX],
           &qm->tdmXgrad[i][YY],
           &qm->tdmXgrad[i][ZZ],
           &qm->tdmYgrad[i][XX],
           &qm->tdmYgrad[i][YY],
           &qm->tdmYgrad[i][ZZ],
           &qm->tdmZgrad[i][XX],
           &qm->tdmZgrad[i][YY],
           &qm->tdmZgrad[i][ZZ]);
#endif
    /* compile the gradients of mu.E wrt the atom positions */

    }
    /* pick up ref TDM at the last line of the file */
    if(NULL == fgets(buf,3000,in)){
        gmx_fatal(FARGS,"Error reading TDM gradient file");
    }
#ifdef GMX_DOUBLE
    sscanf(buf,"%lf %lf %lf\n",
           &qm->tdm[XX],
           &qm->tdm[YY],
           &qm->tdm[ZZ]);
#else
    sscanf(buf,"%f %f %f\n",
           &qm->tdm[XX],
           &qm->tdm[YY],
           &qm->tdm[ZZ]);
#endif
    fclose(in);
//    free(buf);
    free (reffile);
    free(reftitle);
    free(tdmfile);
  }
  /* Hardcode... */   
  snew(mass,100);
  mass[1]=1.008;
  mass[6]=12.01;
  mass[7]=14.01;
  mass[8]=16;
  snew(w_rls,qm->nrQMatoms);
  snew(xcopy,qm->nrQMatoms);
   
  for (i=0;i<qm->nrQMatoms;i++){
    
    mm=mass[qm->atomicnumberQM[i]];
    tm+=mm;
    w_rls[i]=mm;
    for(j=0;j<DIM;j++){
      xrefcm[j]+= mm*qm->xref[i][j];
      xcm[j]+= mm*qm->xQM[i][j];
      xcopy[i][j]=qm->xQM[i][j];
    }
  }
  for(j=0;j<DIM;j++){
    xcm[j] /= tm;
    xrefcm[j] /= tm;
  }
  for(i=0;i<qm->nrQMatoms;i++){
    rvec_dec(xcopy[i],xcm);
    rvec_dec(qm->xref[i],xrefcm);
  }
//  fprintf(stderr,"x[0]    : %lf,%lf%lf\n",qm->xQM[0][0],qm->xQM[0][1],qm->xQM[0][2]);
//  fprintf(stderr,"xcopy[0]: %lf,%lf%lf\n",xcopy[0][0],xcopy[0][1],xcopy[0][2]);
  calc_fit_R(3,qm->nrQMatoms,w_rls,qm->xref,xcopy,R);
  transpose(R,Rinv);
  /* rotate the field */
  for (j=0;j<DIM;j++){
    Erot[j]=0;
    for(k=0;k<DIM;k++){
      Erot[j]+=R[j][k]*E[k];
    }
  }
//  fprintf(stderr, "rotated field: %lf %lf %lf\n",Erot[XX],Erot[YY],Erot[ZZ]);
  /* calculate the gradient in the rotated frame */
  snew(tdmgrad_rot,qm->nrQMatoms);
  for(i=0;i<qm->nrQMatoms;i++){
    for(j=0;j<DIM;j++){
      tdmgrad_rot[i][j]=qm->tdmXgrad[i][j]*Erot[XX]+qm->tdmYgrad[i][j]*Erot[YY]+qm->tdmZgrad[i][j]*Erot[ZZ];
    }
  }
  /* calculate the TDM in the rotated field */
  snew(xfit,qm->nrQMatoms);
  for (j=0;j<qm->nrQMatoms;j++){
    for(k=0;k<DIM;k++){
      
      xfit[j][k]=0;
      for (l=0;l<DIM;l++){
        xfit[j][k]+=R[k][l]*xcopy[j][l];
      }
    }
  }
//  fprintf(stderr,"xfit[0] : %lf,%lf,%lf\n",xfit[0][0],xfit[0][1],xfit[0][2]);
//  fprintf(stderr,"rref[0] : %lf,%lf,%lf\n",qm->xref[0][0],qm->xref[0][1],qm->xref[0][2]);
  for(k=0;k<DIM;k++){
   tdm[k]=qm->tdm[k];
  }
  for(j=0;j<qm->nrQMatoms;j++){
    for(k=0;k<DIM;k++){
      tdm[XX]+=qm->tdmXgrad[j][k]*(xfit[j][k]-qm->xref[j][k]);
      tdm[YY]+=qm->tdmYgrad[j][k]*(xfit[j][k]-qm->xref[j][k]);
      tdm[ZZ]+=qm->tdmZgrad[j][k]*(xfit[j][k]-qm->xref[j][k]);
    }
  }
  free(xfit);
  /* rotate back the TDM */
  rvec tdm_R;
  for (j=0;j<DIM;j++){
    for (k=0;k<DIM;k++){
      tdm_R[j]+=Rinv[j][k]*tdm[k];
    }
  }
  for (j=0;j<DIM;j++){
    tdm[j]=tdm_R[j];
  }


  /* rotate back the gradients */
  for (i=0;i<qm->nrQMatoms;i++){
    for (j=0;j<DIM;j++){
      tdmgrad[i][j]=0;
      for(k=0;k<DIM;k++){
        tdmgrad[i][j]+=Rinv[j][k]*tdmgrad_rot[i][k];
      }
    }
  }
  free(tdmgrad_rot);
  /* I suppose there should be another gradient to try and allign dipole
   */
  free(w_rls);
  free(xcopy);
  free(mass);
  step++;
} /* get_dipole_gradients */

void   propagate_TDSE(int step, t_QMrec *qm, dplx *eigvec, int ndim, double *eigval, real dt, t_QMMMrec *qr){
  fprintf(stderr,"Call to propagate_TDSE\n");
  int
    i;
  dplx 
    *c=NULL,*cold=NULL,*U=NULL; 
//  FILE
//    *Cout=NULL;
//  char
//    buf[5000];

  snew(c,ndim); 
  snew(cold,ndim);

  if(step){
    for (i=0;i<ndim;i++){
      cold[i]=qm->creal[i]+IMAG*qm->cimag[i];
      c[i]=qm->creal[i]+IMAG*qm->cimag[i];
//      fprintf(stderr,"|c[%d]|^2 = %lf ;",i,conj(c[i])*c[i]);
//      fprintf(stderr,"|g[%d]|^2 = %lf\n",i,conj(cold[i])*cold[i]);
    }
    track_states(qm->eigvec, eigvec, ndim);
    /* we propagate the wave function in the local diabatic basis, i.e.
     * diabatic along the direction in which the atoms moved.
     * J. CHem. Phys. 114 (2001) 10608  
     *
     */
    snew(U,ndim*ndim);
    propagate_local_dia(ndim,dt,c,eigvec,qm->eigvec,eigval,qm->eigval,U);
    for ( i = 0 ; i < ndim ; i++ ){
      qm->creal[i] = creal(c[i]);
      qm->cimag[i] = cimag(c[i]);
    } 
    /* some writinig
     */
    fprintf(stderr,"step %d: C: ",step);
//    sprintf(buf,"%s/C.dat",qm->work_dir);
//    Cout=fopen (buf,"w");
//    fprintf(Cout,"%d\n",step);
    for(i=0;i<ndim;i++){
      fprintf (stderr," %.5lf ",conj(c[i])*c[i]);    
//      fprintf (Cout,"%.5lf %.5lf\n ",qm->creal[i],qm->cimag[i]);
    }
//    fclose(Cout);
    fprintf(stderr,"\n");
    free(U);
  }
  else{
    if(qm->QEDrestart){
      track_states(qm->eigvec,eigvec,ndim);
    }
    //qm->creal[qm->polariton]=1.0;
    fprintf(stderr,"step %d: C: ",step);
    for(i=0;i<ndim;i++){
      c[i] = qm->creal[i]+ IMAG*qm->cimag[i];
      fprintf (stderr,"%.5lf ",conj(c[i])*c[i]);
    }    
    fprintf(stderr,"\n");
  }
  free(c);
  free(cold);
  fprintf(stderr,"propagate_TDSE done\n");
}

double cavity_dispersion(int n, t_QMrec *qm){
  return sqrt(qm->omega*qm->omega+SPEED_OF_LIGHT_AU*SPEED_OF_LIGHT_AU*(2*M_PI*n/(qm->L*microM2BOHR))*(2*M_PI*n/(qm->L*microM2BOHR))/(qm->n_index*qm->n_index));
} /* cavity_dispersion */

#ifdef nacs_flag
/* Print the non-adiabatic coupling vectors HF_f */
void dump_nacs(int m, int nmol, int ndim, double *eigval, dplx *eigvec, double *u, rvec *tdmX,
		 rvec *tdmY, rvec *tdmZ, rvec *QMgrad_S0, rvec *QMgrad_S1, t_QMrec *qm, t_forcerec *fr){
  int 
    i,j,n,p,q,begin,end,a;
  double 
    V0_2EP;
  dplx 
    fij,betasq,a_sum,a_sumq;  
  dplx
    bp_aq,ap_bq;
  rvec
    *HF_f;
  char
    *non_adiab_couplings;
  FILE
    *nacs=NULL;

  V0_2EP = qm->omega/iprod(qm->E,qm->E); // Cavity volume at zero k

  if(fr->qr->SHmethod != eSHmethodEhrenfest){ /* Single state procedure  */
    begin=qm->polariton;    
    end=qm->polariton+1;
  }
  else{
    /* Ehrenfest dynamics, need to compute gradients of all polaritonic states 
     * and weight them with weights of the states. Also the nonadiabatic couplings 
     * between polaritonic states are needed now */
    begin=0;
    end=ndim;
  }

  /* Memory allocation of non-adiabatic couplings string & vector HF_f */
  snew(non_adiab_couplings,3000);
  snew(HF_f,qm->nrQMatoms);

  for (p=begin;p<end;p++){
    /* diagonal terms are zero but we need a_sum=a_sump */
    n=qm->n_min;
    a_sum = 0.0+IMAG*0.0;
    for (i=0;i<(qm->n_max-qm->n_min+1);i++){
      a_sum += eigvec[p*ndim+nmol+i]*sqrt(cavity_dispersion(n++,qm)/V0_2EP)*cexp(+IMAG*2*M_PI*(n-1)*qm->z[m]/(qm->L*microM2BOHR));
    }

    /* now off-diagonals */
    for (q=p+1;q<end;q++){
      betasq = conj(eigvec[p*ndim+m])*eigvec[q*ndim+m];

      n=qm->n_min;
      a_sumq = 0.0+IMAG*0.0;
      for (i=0;i<(qm->n_max-qm->n_min+1);i++){
	a_sumq += eigvec[q*ndim+nmol+i]*sqrt(cavity_dispersion(n++,qm)/V0_2EP)*cexp(+IMAG*2*M_PI*(n-1)*qm->z[m]/(qm->L*microM2BOHR));
      }

      bp_aq = conj(eigvec[p*ndim+m])*a_sumq;	// (beta_p)*(sum_of alphas_q
      ap_bq = eigvec[q*ndim+m]*conj(a_sum); 

      /* Non-adiabatic couplings file nacs_m_p-q.dat (m=molecule & p,q=polaritons) */
      sprintf(non_adiab_couplings,"%s/nacs_%d_%d-%d.dat",qm->work_dir,m,p,q);
      nacs=fopen(non_adiab_couplings,"w");
      fprintf(nacs,"%12.8lf %12.8lf\n",eigval[q]*HARTREE2KJ*AVOGADRO,eigval[p]*HARTREE2KJ*AVOGADRO);
 
      /* Assemble terms to compute non-adiabatic couplings */
      for(i=0;i<qm->nrQMatoms;i++){
	for(j=0;j<DIM;j++){
	  /* diagonal term */
	  fij = (betasq)*(QMgrad_S1[i][j]-QMgrad_S0[i][j]);
	  /* off-diagonal term */
	  fij-= (bp_aq+ap_bq)*tdmX[i][j]*u[0];
	  fij-= (bp_aq+ap_bq)*tdmY[i][j]*u[1];
	  fij-= (bp_aq+ap_bq)*tdmZ[i][j]*u[2];
	  fij*=HARTREE_BOHR2MD;

	  /* NACs vector */
	  HF_f[i][j]=creal(fij);
	}
      }
      /* Print non-adiabatic couplings per atom */
      for(a=0;a<qm->nrQMatoms;a++){
	fprintf(nacs,"%12.8lf %12.8lf %12.8lf\n",HF_f[a][0],HF_f[a][1],HF_f[a][2]);
      }
      fclose(nacs);
    };
  };
  free(non_adiab_couplings);
  free(HF_f);
  return;
} /* Print non-adiabatic coupling vectors */
#endif

/* Compute Hellman Feynman forces */
double HF_forces(int m, int nmol, int ndim, double *eigval, dplx *eigvec, double *u, rvec *tdmX,
		 rvec *tdmY, rvec *tdmZ, rvec *tdmXMM, rvec *tdmYMM, rvec *tdmZMM, rvec *QMgrad_S0,
		 rvec *MMgrad_S0, rvec *QMgrad_S1, rvec *MMgrad_S1, rvec f[], rvec fshift[], 
		 t_QMrec *qm, t_MMrec *mm, t_forcerec *fr){
  int 
    i,j,n,p,q,begin,end,other_m; //other molecules than that corresponding to m
  double 
    V0_2EP,QMener;
  dplx 
    fij,betasq,betasq2,a_sum,a_sumq,csq,csq2,asq_bsq;  
  dplx
    ap_bp,bp_ap,bp_aq,aq_bp,ap_bq,bq_ap;

  V0_2EP = qm->omega/iprod(qm->E,qm->E); // Cavity volume at zero k
  QMener=0.0; 

  if(fr->qr->SHmethod != eSHmethodEhrenfest){ /* Single state procedure  */
    begin=qm->polariton;    
    end=qm->polariton+1;
  }
  else{
    /* Ehrenfest dynamics, need to compute gradients of all polaritonic states 
     * and weight them with weights of the states. Also the nonadiabatic couplings 
     * between polaritonic states are needed now */
    begin=0;
    end=ndim;
  }

  for (p=begin;p<end;p++){
    /* do the diagonal terms first p=q. These terms are same as for 
     * single state procedred */
    if(fr->qr->SHmethod != eSHmethodEhrenfest){
      csq=1.0+IMAG*0.0;
    }
    else{
      csq = conj(qm->creal[p]+IMAG*qm->cimag[p])*(qm->creal[p]+IMAG*qm->cimag[p]);
    }

    betasq = conj(eigvec[p*ndim+m])*eigvec[p*ndim+m];

    asq_bsq=0.0+IMAG*0.0; //Which is actually same as (1-betasq)
    for (other_m=0;other_m<nmol;other_m++){
      if(other_m!=m){
	asq_bsq += conj(eigvec[p*ndim+other_m])*eigvec[p*ndim+other_m];
      }
    } 

    n=qm->n_min;
    a_sum = 0.0+IMAG*0.0;
    for (i=0;i<(qm->n_max-qm->n_min+1);i++){
      a_sum += eigvec[p*ndim+nmol+i]*sqrt(cavity_dispersion(n++,qm)/V0_2EP)*cexp(+IMAG*2*M_PI*(n-1)*qm->z[m]/(qm->L*microM2BOHR));
      asq_bsq += conj(eigvec[p*ndim+nmol+i])*eigvec[p*ndim+nmol+i];
    }
    bp_ap = conj(eigvec[p*ndim+m])*a_sum;	// (beta_p)*(sum_of alphas_p)
    ap_bp = eigvec[p*ndim+m]*conj(a_sum);	//(sum_of alphas_p)*beta_p

    for(i=0;i<qm->nrQMatoms;i++){
      for(j=0;j<DIM;j++){
	/* diagonal term */
	//fij =(betasq*QMgrad_S1[i][j]+(1-betasq)*QMgrad_S0[i][j]);
	fij =(betasq*QMgrad_S1[i][j]+asq_bsq*QMgrad_S0[i][j]);
	/* off-diagonal term, Because coeficients are real: ab = ba*/
	fij-= (bp_ap+ap_bp)*tdmX[i][j]*u[0];
	fij-= (bp_ap+ap_bp)*tdmY[i][j]*u[1];
	fij-= (bp_ap+ap_bp)*tdmZ[i][j]*u[2];
	fij*=HARTREE_BOHR2MD*csq;

	f[i][j]      += creal(fij);
	fshift[i][j] += creal(fij);
      }
    }
    for(i=0;i<mm->nrMMatoms;i++){
      for(j=0;j<DIM;j++){
	/* diagonal terms */
	fij =(betasq*MMgrad_S1[i][j]+asq_bsq*MMgrad_S0[i][j]);
	/* off-diagonal term */
	fij-= (bp_ap+ap_bp)*tdmXMM[i][j]*u[0];
	fij-= (bp_ap+ap_bp)*tdmYMM[i][j]*u[1];
	fij-= (bp_ap+ap_bp)*tdmZMM[i][j]*u[2];
	fij*=HARTREE_BOHR2MD*csq;
          
	f[i+qm->nrQMatoms][j]      += creal(fij);
	fshift[i+qm->nrQMatoms][j] += creal(fij);
      }
    }    
    QMener += csq*eigval[p]*HARTREE2KJ*AVOGADRO;

    /* now off-diagonals */
    for (q=p+1;q<end;q++){
      csq = conj(qm->creal[p]+IMAG*(qm->cimag[p]))*(qm->creal[q]+IMAG*(qm->cimag[q]));
      csq2 = (qm->creal[p]+IMAG*(qm->cimag[p]))*conj(qm->creal[q]+IMAG*(qm->cimag[q]));
      betasq = conj(eigvec[p*ndim+m])*eigvec[q*ndim+m];
      betasq2 = eigvec[p*ndim+m]*conj(eigvec[q*ndim+m]);

      n=qm->n_min;
      a_sumq = 0.0+IMAG*0.0;
      for (i=0;i<(qm->n_max-qm->n_min+1);i++){
	a_sumq += eigvec[q*ndim+nmol+i]*sqrt(cavity_dispersion(n++,qm)/V0_2EP)*cexp(+IMAG*2*M_PI*(n-1)*qm->z[m]/(qm->L*microM2BOHR));
      }

      bp_aq = conj(eigvec[p*ndim+m])*a_sumq;	// (beta_p)*(sum_of alphas_q
      aq_bp = eigvec[p*ndim+m]*conj(a_sumq);	// (sum_of alhas_q)*beta_p
      bq_ap = conj(eigvec[q*ndim+m])*a_sum; 
      ap_bq = eigvec[q*ndim+m]*conj(a_sum); 

      for(i=0;i<qm->nrQMatoms;i++){
	for(j=0;j<DIM;j++){
	  /* diagonal term */
	  fij = (csq*betasq+csq2*betasq2)*(QMgrad_S1[i][j]-QMgrad_S0[i][j]);
	  /* off-diagonal term */
	  fij-= ((csq*(bp_aq+ap_bq)+csq2*(aq_bp+bq_ap))*tdmX[i][j]*u[0]);
	  fij-= ((csq*(bp_aq+ap_bq)+csq2*(aq_bp+bq_ap))*tdmY[i][j]*u[1]);
	  fij-= ((csq*(bp_aq+ap_bq)+csq2*(aq_bp+bq_ap))*tdmZ[i][j]*u[2]);
	  fij*=HARTREE_BOHR2MD;

	  f[i][j]      += creal(fij);
	  fshift[i][j] += creal(fij);
	}
      }
      for(i=0;i<mm->nrMMatoms;i++){
	for(j=0;j<DIM;j++){
	  /* diagonal term */
	  fij = (csq*betasq+csq2*betasq2)*(MMgrad_S1[i][j]-MMgrad_S0[i][j]);
	  /* off-diagonal term */
	  fij-= ((csq*(bp_aq+ap_bq)+csq2*(aq_bp+bq_ap))*tdmXMM[i][j]*u[0]);
	  fij-= ((csq*(bp_aq+ap_bq)+csq2*(aq_bp+bq_ap))*tdmYMM[i][j]*u[1]);
	  fij-= ((csq*(bp_aq+ap_bq)+csq2*(aq_bp+bq_ap))*tdmZMM[i][j]*u[2]);
	  fij*=HARTREE_BOHR2MD;

	  f[i+qm->nrQMatoms][j]      += creal(fij);
	  fshift[i+qm->nrQMatoms][j] += creal(fij);
	}
      }
    }
  }
  return QMener;
} /* Hellman Feynman forces */


void decoherence(t_commrec *cr, t_QMrec *qm, t_MMrec *mm, int ndim, double *eigval){
/* decoherence corretion by Grunucci et al (J. Chem. Phys. 126, 134114 (2007) */
  int 
    state;
  double 
    sum,decay=0.0,tau,ekin[1];
  
  ekin[0] = 0.0;
  /* kinetic energy of the qm nuclei and mm pointcharges on each node */
  ekin[0]  = dottrrr(qm->nrQMatoms, qm->ffmass, qm->vQM, qm->vQM);
  ekin[0] += dottrrr(mm->nrMMatoms, mm->ffmass, mm->vMM, mm->vMM);
  ekin[0] *= 0.5*VEL2AU*VEL2AU*MP2AU;
  /* now we have ekin per node. Now send around the total kinetic energy */
  /* send around */
  if(MULTISIM(cr)){
    gmx_sumd_sim(1,ekin,cr->ms);
  }
  /* apply */ 
  sum = 0.0;
  for (state = 0; state < ndim; state++){
    if (state != qm->polariton){
      tau = (1.0+(qm->QEDdecoherence)/ekin[0])/fabs(eigval[state]-eigval[qm->polariton]);
//      if(MULTISIM(cr)){
//        if (cr->ms->sim==0){
//          fprintf(stderr,"node %d: tau  = %lf, exp(-dt/tau)=%lf\n",cr->ms->sim,tau,exp(-(qm->dt)/(tau*AU2PS)));
//        }
//      }
//      else{
//        fprintf(stderr,"tau = %lf, exp(-dt/tau)=%lf\n",tau,exp(-(qm->dt)/(tau*AU2PS)));
//      }
      qm->creal[state]*=exp(-(qm->dt)/(tau*AU2PS));
      qm->cimag[state]*=exp(-(qm->dt)/(tau*AU2PS));
      sum += qm->creal[state]*qm->creal[state]+qm->cimag[state]*qm->cimag[state];
//      if(MULTISIM(cr)){
//        if (cr->ms->sim==0){
//          fprintf(stderr,"node %d: state %d, |c_m|^2=%lf,sum = %lf\n",cr->ms->sim,state,qm->creal[state]*qm->creal[state]+qm->cimag[state]*qm->cimag[state],sum);
//        }
//      }
//      else{
//        fprintf(stderr,"state %d, sum = %lf\n",cr->ms->sim,state,sum);
//      }
    }
  }
//  fprintf(stderr,"sum = %lf, |c_M(0)|^2 = %lf; sum+|c_M(0)|^2=%lf\n",sum,(qm->creal[qm->polariton])*(qm->creal[qm->polariton])+(qm->cimag[qm->polariton])*(qm->cimag[qm->polariton]),sum+(qm->creal[qm->polariton])*(qm->creal[qm->polariton])+(qm->cimag[qm->polariton])*(qm->cimag[qm->polariton]));
  /* add the contribution of the ground state too */
  sum+=qm->groundstate;
  decay = sqrt((1.0-sum)/((qm->creal[qm->polariton])*(qm->creal[qm->polariton])+(qm->cimag[qm->polariton])*(qm->cimag[qm->polariton])));
  qm->creal[qm->polariton]*=decay;
  qm->cimag[qm->polariton]*=decay;
  if(MULTISIM(cr)){
    if (cr->ms->sim==0){
      fprintf(stderr,"node %d: decoherence done, decay = %lf\n",cr->ms->sim,decay);
    }
  }
  else{
    fprintf(stderr,"decoherence done, decay = %lf\n",decay);
  }
} /* decoherence */

real call_gaussian_QED(t_commrec *cr,  t_forcerec *fr, 
		   t_QMrec *qm, t_MMrec *mm, rvec f[], rvec fshift[]){
  /* multiple gaussian jobs for QED */
  static int
    step=0;
  int
    i,j=0,k,m,ndim,nmol;
  double
    QMener=0.0,*energies,Eground,c;
  rvec
    *QMgrad_S0,*MMgrad_S0,*QMgrad_S1,*MMgrad_S1,tdm;
  rvec
    *tdmX,*tdmY,*tdmZ,*tdmXMM,*tdmYMM,*tdmZMM;
  char
    *exe,*eigenvectorfile,*final_eigenvecfile,*energyfile,buf[3000];
  double
    *eigval,*tmp=NULL,L_au=qm->L*microM2BOHR,decay,asq;
  dplx
    *eigvec,*matrix=NULL,*couplings=NULL;
  double
    *eigvec_real,*eigvec_imag,*send_couple;
  int
    dodia=1,*state,p,q;
  FILE
    *evout=NULL,*final_evout=NULL,*enerout=NULL,*Cout=NULL;
  time_t 
    start,end,interval;

  start = time(NULL);
  snew(exe,3000);
  sprintf(exe,"%s/%s",qm->gauss_dir,qm->gauss_exe);
  snew(eigenvectorfile,3000);
  sprintf(eigenvectorfile,"%s/eigenvectors.dat",qm->work_dir);
  snew(final_eigenvecfile,3000);
  sprintf(final_eigenvecfile,"%s/final_eigenvecs.dat",qm->work_dir);

  /*  excited state forces */
  snew(QMgrad_S1,qm->nrQMatoms);
  snew(MMgrad_S1,mm->nrMMatoms);

  /* ground state forces */
  snew(QMgrad_S0,qm->nrQMatoms);
  snew(MMgrad_S0,mm->nrMMatoms);
  snew(tdmX,qm->nrQMatoms);
  snew(tdmY,qm->nrQMatoms);
  snew(tdmZ,qm->nrQMatoms);
  snew(tdmXMM,mm->nrMMatoms);
  snew(tdmYMM,mm->nrMMatoms);
  snew(tdmZMM,mm->nrMMatoms);
  write_gaussian_input_QED(cr,step,fr,qm,mm);

  /* silly array to communicate the courrent state*/
  snew(state,1);

  /* we use the script to use QM code */ 
  do_gaussian(step,exe);
  interval=time(NULL);
  if (MULTISIM(cr)){
    if (cr->ms->sim==0)
      fprintf(stderr,"node %d: do_gaussian done at %ld\n",cr->ms->sim,interval-start);
  }
  else{
    fprintf(stderr,"node 0: read_gaussian done at %ld\n",interval-start);
  }
  QMener = read_gaussian_output_QED(cr,QMgrad_S1,MMgrad_S1,QMgrad_S0,MMgrad_S0,
				    step,qm,mm,&tdm,tdmX,tdmY,tdmZ,
                                    tdmXMM,tdmYMM,tdmZMM,&Eground);
  interval=time(NULL);
  if(MULTISIM(cr)){
    if (cr->ms->sim==0){
      fprintf(stderr,"node %d: read_gaussian done at %ld\n",cr->ms->sim,interval-start);
    }
    ndim=cr->ms->nsim+qm->n_max-qm->n_min+1;
    m=cr->ms->sim;
    nmol=cr->ms->nsim;
  }
  else{
    fprintf(stderr,"read_gaussian done at %ld\n",interval-start);
    ndim=1+qm->n_max-qm->n_min+1;
    fprintf(stderr,"ndim= %d\n",ndim);
    m=0;
    nmol=1;
  }
  snew(energyfile,3000);
  sprintf(energyfile,"%s/%s%d.dat",qm->work_dir,"energies",m);
  enerout=fopen(energyfile,"a");
  fprintf(enerout,"step %d E(S0): %12.8lf E(S1) %12.8lf TDM: %12.8lf %12.8lf %12.8lf\n",step,Eground, QMener, tdm[XX],tdm[YY],tdm[ZZ]);
  fclose(enerout);

  snew(energies,ndim);
  /* on the diagonal there is the excited state energy of the molecule
   * plus the ground state energies of all other molecules
   */
  for (i=0;i<ndim;i++){
    energies[i]=Eground;
  }
  energies[m]=QMener; /* the excited state energy, overwrites
			 the ground state energy */
  /* send around */
  if(MULTISIM(cr)){
    gmx_sumd_sim(ndim,energies,cr->ms);
  }
  for (i=0;i<(qm->n_max-qm->n_min+1);i++){
    energies[nmol+i]+=cavity_dispersion(qm->n_min+i,qm);
  } /* after summing the ground state energies, the photon energy of the cavity 
      (such that w[k]=w[-k]) is added to the last (n_max-n_min+1) diagonal terms */

  /* now we fill up the off-diagonals, basically the dot product of the dipole
     moment with the unit vector of the E-field of the cavity/plasmon times the
     E-field magnitud that is now k-dependent through w(k)
  */
  snew(couplings,nmol*(qm->n_max-qm->n_min+1));
  double V0_2EP = qm->omega/iprod(qm->E,qm->E); //2*Epsilon0*V_cav at k=0 (in a.u.)
  double u[3];
  u[0]=qm->E[0]/sqrt(iprod(qm->E,qm->E));
  u[1]=qm->E[1]/sqrt(iprod(qm->E,qm->E));
  u[2]=qm->E[2]/sqrt(iprod(qm->E,qm->E)); //unit vector in E=Ex*u1+Ey*u2+Ez*u3
  for (i=0;i<(qm->n_max-qm->n_min+1);i++){
    couplings[m*(qm->n_max-qm->n_min+1)+i] = -iprod(tdm,u)*sqrt(cavity_dispersion(qm->n_min+i,qm)/V0_2EP)*cexp(+IMAG*2*M_PI*(qm->n_min+i)/L_au*qm->z[m]);
  }
  /* send couplings around */
  snew(send_couple,2*nmol*(qm->n_max-qm->n_min+1));
  for (i=0;i<nmol*(qm->n_max-qm->n_min+1);i++){
    send_couple[i]=creal(couplings[i]);
    send_couple[nmol*(qm->n_max-qm->n_min+1)+i]=cimag(couplings[i]);
  }
  if(MULTISIM(cr)){
    gmx_sumd_sim(2*nmol*(qm->n_max-qm->n_min+1),send_couple,cr->ms);
  }
  for (i=0;i<nmol*(qm->n_max-qm->n_min+1);i++){
    couplings[i]=send_couple[i]+IMAG*send_couple[nmol*(qm->n_max-qm->n_min+1)+i];
  }

  /* diagonalize the QED matrix on the masternode, and communicate back
     the eigenvectors to compute the Hellman Feynman forces
  */
  snew(eigval,ndim);
  snew(eigvec,ndim*ndim);
  snew(eigvec_real,ndim*ndim);
  snew(eigvec_imag,ndim*ndim);

  if(MULTISIM(cr)){
    if (!MASTERSIM(cr->ms)){
      dodia = 0;
    }
  }
  interval=time(NULL);
  if(MULTISIM(cr)){
    if (cr->ms->sim==0)
      fprintf(stderr,"node %d: matrix build at at %ld\n",cr->ms->sim,interval-start);
  }
  else{
    fprintf(stderr,"Matrix build at %ld\n",interval-start);
  }
  if(dodia){
    snew(matrix,ndim*ndim);
    for (i=0;i<ndim;i++){
      matrix[i+(i*ndim)]=energies[i];
    }
    for (k=0;k<nmol;k++){
      for (j=0;j<(qm->n_max-qm->n_min+1);j++){
	matrix[nmol+j+(k*ndim)]= conj(couplings[k*(qm->n_max-qm->n_min+1)+j]);
	matrix[ndim*nmol+k+(j*ndim)]= (couplings[k*(qm->n_max-qm->n_min+1)+j]);
      }
    }

    fprintf(stderr,"\n\ndiagonalizing matrix\n");
    diag(ndim,eigval,eigvec,matrix);
    fprintf(stderr,"step %d Eigenvalues: ",step);
    for(i=0;i<ndim;i++){
      fprintf(stderr,"%lf ",eigval[i]);
      qm->eigval[i]=eigval[i]; 
    }
    fprintf(stderr,"\n");
    free(matrix);
    interval=time(NULL);
    if(MULTISIM(cr)){
      fprintf(stderr,"node %d: eigensolver done at %ld\n",cr->ms->sim,interval-start);
    }
    else{
      fprintf(stderr,"eigensolver done at %ld\n",interval-start);
    }

    if(fr->qr->SHmethod != eSHmethodEhrenfest){
      if(fr->qr->SHmethod == eSHmethoddiabatic){
	qm->polariton = QEDhop(step,qm,eigvec,ndim,eigval);
      }
      else{
	qm->polariton = QEDFSSHop(step,qm,eigvec,ndim,eigval,qm->dt,fr->qr);
      }
      state[0]=qm->polariton;
    } 
    else{
      propagate_TDSE(step,qm,eigvec,ndim,eigval,qm->dt,fr->qr);
    } 
    interval=time(NULL);
    if(MULTISIM(cr)){
      fprintf(stderr,"node %d: wavefunction propagation done at %ld\n",cr->ms->sim,interval-start); 
    }
    else{
      fprintf(stderr,"wavefunction propagation done at %ld\n",interval-start);
    }
  }
  else{/* zero the expansion coefficient on all other nodes */
    free(qm->cimag);
    free(qm->creal);
    snew(qm->cimag,ndim);
    snew(qm->creal,ndim);
  }  
  /* send the eigenvalues and eigenvectors around */
  for(i=0;i<ndim*ndim;i++){
    eigvec_real[i]=creal(eigvec[i]);
    eigvec_imag[i]=cimag(eigvec[i]);
  }
  if(MULTISIM(cr)){
    gmx_sumd_sim(ndim*ndim,eigvec_real,cr->ms);
    gmx_sumd_sim(ndim*ndim,eigvec_imag,cr->ms);
    gmx_sumd_sim(ndim,eigval,cr->ms);
    if(fr->qr->SHmethod != eSHmethodEhrenfest){
      gmx_sumi_sim(1,state,cr->ms);
      qm->polariton = state[0];
    }
    /* communicate the time-dependent expansion coefficients needed for computing mean-field forces */
    gmx_sumd_sim(ndim,qm->creal ,cr->ms);
    gmx_sumd_sim(ndim,qm->cimag ,cr->ms);
  }
  for(i=0;i<ndim*ndim;i++){
    eigvec[i]=eigvec_real[i]+IMAG*eigvec_imag[i];
  }
  interval=time(NULL);
  if(MULTISIM(cr)){
    if (cr->ms->sim==0)
      fprintf(stderr,"node %d: gmx_sumd_sim  done at %ld\n",cr->ms->sim,interval-start); 
  } 
  else{
    fprintf(stderr,"node 0: gmx_sumd_sim done at %ld\n",interval-start);
  }
  /* copy the eigenvectors to qmrec */
  for(i=0;i<ndim*ndim;i++){
    qm->eigvec[i]=eigvec[i];
  }

  /* compute Hellman Feynman forces */
  QMener = HF_forces(m,nmol,ndim,eigval,eigvec,u,tdmX,tdmY,tdmZ,tdmXMM,tdmYMM,
		     tdmZMM,QMgrad_S0,MMgrad_S0,QMgrad_S1,MMgrad_S1,f,fshift,qm,mm,fr);
#ifdef nacs_flag
  dump_nacs(m,nmol,ndim,eigval,eigvec,u,tdmX,tdmY,tdmZ,QMgrad_S0,QMgrad_S1,qm,fr);
#endif
    
  interval=time(NULL);
  if(MULTISIM(cr)){
    if (cr->ms->sim==0) 
      fprintf(stderr,"node %d: Forces done at %ld\n",cr->ms->sim,interval-start);
  }
  else{
    fprintf(stderr,"Forces done at %ld\n",interval-start);
  }

//  if ( qm->QEDdecay > 0 ){
  if ( fr->qr->SHmethod == eSHmethodEhrenfest ){    
    QMener+=(qm->groundstate)*(energies[ndim-1]-cavity_dispersion(qm->n_max,qm));
    for(i=0;i<qm->nrQMatoms;i++){
      for(j=0;j<DIM;j++){
        f[i][j]      += (qm->groundstate)*HARTREE_BOHR2MD*QMgrad_S0[i][j];
        fshift[i][j] += (qm->groundstate)*HARTREE_BOHR2MD*QMgrad_S0[i][j];
      }
    }
    for(i=0;i<mm->nrMMatoms;i++){
      for(j=0;j<DIM;j++){
        f[i+qm->nrQMatoms][j]      += qm->groundstate*HARTREE_BOHR2MD*MMgrad_S0[i][j];
        fshift[i+qm->nrQMatoms][j] += qm->groundstate*HARTREE_BOHR2MD*MMgrad_S0[i][j];
      }
    }
  }

  /* Print the eigenvectors to the eigenvectors.dat file  */
  if((dodia) ){
    evout=fopen(eigenvectorfile,"a");
    if ((fr->qr->SHmethod != eSHmethodEhrenfest) ){
      i = qm->polariton;
      fprintf(evout,"step %d Eigenvector %d:",step,i);
      for (k=0;k<ndim;k++){
        fprintf(evout," %12.8lf + %12.8lf I  ",creal(eigvec[i*ndim+k]),cimag(eigvec[i*ndim+k]));
      }
      fprintf(evout,"\n");
      int ii;
      final_evout=fopen(final_eigenvecfile,"a");
      for(ii=0;ii<ndim;ii++){
	fprintf(final_evout,"step %d Eigenvector %d gap %lf (c: %12.8lf + %12.8lf I):",step,ii,eigval[ii]-(qm->groundstate,energies[ndim-1]-cavity_dispersion(qm->n_max,qm)),qm->creal[ii],qm->cimag[ii]);
	for(k=0;k<ndim;k++){
	  fprintf(final_evout," %12.8lf + %12.8lf I ",creal(eigvec[ii*ndim+k]),cimag(eigvec[ii*ndim+k]));
	}
      fprintf(final_evout,"\n");
      }
      fclose(final_evout);
    }
    else{
      /* Ehrenfest, print all vectors
       * Thee will need to be multiplied by the coefficients. 
       */
      for(i=0;i<ndim;i++){
	fprintf(evout,"step %d Eigenvector %d gap %lf (c: %12.8lf + %12.8lf I):",step,i,eigval[i]-(qm->groundstate,energies[ndim-1]-cavity_dispersion(qm->n_max,qm)),qm->creal[i],qm->cimag[i]);
	for(k=0;k<ndim;k++){
	  fprintf(evout," %12.8lf + %12.8lf I ",creal(eigvec[i*ndim+k]),cimag(eigvec[i*ndim+k]));
	}
	fprintf(evout,"\n");
      }
    }
    fclose(evout);
    interval=time(NULL);
    if(MULTISIM(cr)){
      fprintf(stderr,"node %d: printing eigenvectors done at %ld\n",cr->ms->sim,interval-start);
    }
    else{
      fprintf(stderr,"printing eigenvectors done at %ld\n",interval-start);
    }
  }

  if ( fr->qr->SHmethod != eSHmethoddiabatic){
    /* printing the coefficients to C.dat */
    if (dodia){
      fprintf(stderr,"rho0 (%d) = %lf, Energy = %lf\n",step,qm->groundstate,energies[ndim-1]-cavity_dispersion(qm->n_max,qm));
      sprintf(buf,"%s/C.dat",qm->work_dir);
      Cout= fopen(buf,"w");
      fprintf(Cout,"%d\n",step);
      for(i=0;i<ndim;i++){
        fprintf (Cout,"%12.8lf %12.8lf\n ",qm->creal[i],qm->cimag[i]);
      }
      fprintf (Cout,"%.5lf\n",qm->groundstate);
      fclose(Cout);
    }
    /* now account for the decay that will happen in the next timestep */
    if (qm->QEDdecay > 0.0){
      for ( i = 0 ; i < ndim ; i++ ){
	asq = 0.0;
	for (j=0;j<(qm->n_max-qm->n_min+1);j++){
	  asq += conj(eigvec[i*ndim+nmol+j])*eigvec[i*ndim+nmol+j];
	}
	decay = exp(-0.5*(qm->QEDdecay)*asq*(qm->dt)); 
	qm->groundstate-=conj(qm->creal[i]+IMAG*qm->cimag[i])*(qm->creal[i]+IMAG*qm->cimag[i])*(decay*decay-1);
	qm->creal[i] *= decay;
	qm->cimag[i] *= decay;
      }
    }
    /* now account also for the decoherence that will also happen in the next timestep */
    /* we thus use the current total kinetic energy. We need to send this around we wrote separate routine */
    if(fr->qr->SHmethod == eSHmethodTully || fr->qr->SHmethod == eSHmethodGranucci){
      decoherence(cr,qm,mm,ndim,eigval);
    }
  }
  interval=time(NULL);
  if(MULTISIM(cr)){
    if (cr->ms->sim==0)
      fprintf(stderr,"node %d: decay done at %ld\n",cr->ms->sim,interval-start);
  }
  else{
    fprintf(stderr,"decay done at %ld\n",interval-start);
  }
//  }
  step++;
  free(exe);
  free(eigenvectorfile);
  free(energyfile);
  free(MMgrad_S0);
  free(QMgrad_S1);
  free(MMgrad_S1);
  free(QMgrad_S0);
  free(tdmX);
  free(tdmY);
  free(tdmZ);
  free(tdmXMM);
  free(tdmYMM);
  free(tdmZMM);
  free(couplings);
  free(send_couple);
  free(eigval);
  free(eigvec);
  free(eigvec_real);
  free(eigvec_imag);
  free(energies);
  free(state);
  free (final_eigenvecfile);
  return(QMener);
} /* call_gaussian_QED */


real call_gaussian_SH(t_commrec *cr, t_forcerec *fr, t_QMrec *qm, t_MMrec *mm, 
		      rvec f[], rvec fshift[])
{ 
  /* a gaussian call routine intended for doing diabatic surface
   * "sliding". See the manual for the theoretical background of this
   * TSH method.  
   */
  static int
    step=0;
  int
    state,i,j;
  real
    QMener=0.0;
  static  gmx_bool
    swapped=FALSE; /* handle for identifying the current PES */
  gmx_bool
    swap=FALSE; /* the actual swap */
  rvec
    *QMgrad,*MMgrad;
  char
    *buf;
  char
    *exe;
  real
    deltaE = 0.0;
  
  snew(exe,30);
  sprintf(exe,"%s/%s",qm->gauss_dir,qm->gauss_exe);
  /* hack to do ground state simulations */
  if(!step){
    snew(buf,20);
    buf = getenv("STATE");
    if (buf)
      sscanf(buf,"%d",&state);
    else
      state=2;
    if(state==1)
      swapped=TRUE;
  }
  /* end of hack */

  /* copy the QMMMrec pointer */
  snew(QMgrad,qm->nrQMatoms);
  snew(MMgrad,mm->nrMMatoms);
  /* at step 0 there should be no SA */
  /*  if(!step)
   * qr->bSA=FALSE;*/
  /* temporray set to step + 1, since there is a chk start */
  write_gaussian_SH_input(step,swapped,fr,qm,mm);

  do_gaussian(step,exe);
  QMener = read_gaussian_SH_output(QMgrad,MMgrad,step,swapped,qm,mm,&deltaE);

  /* check for a surface hop. Only possible if we were already state
   * averaging.
   */
  if(qm->SAstep>0 && deltaE < 0.01){
    if(!swapped){
      swap    = (step && hop(step,qm));
      swapped = swap;
    } 
    else { /* already on the other surface, so check if we go back */
      swap    = (step && hop(step,qm));
      swapped =!swap; /* so swapped shoud be false again */
    }
    if (swap){/* change surface, so do another call */
      write_gaussian_SH_input(step,swapped,fr,qm,mm);
      do_gaussian(step,exe);
      QMener = read_gaussian_SH_output(QMgrad,MMgrad,step,swapped,qm,mm,&deltaE);
    }
  }
  /* add the QMMM forces to the gmx force array and fshift
   */
  for(i=0;i<qm->nrQMatoms;i++){
    for(j=0;j<DIM;j++){
      f[i][j]      = HARTREE_BOHR2MD*QMgrad[i][j];
      fshift[i][j] = HARTREE_BOHR2MD*QMgrad[i][j];
    }
  }
  for(i=0;i<mm->nrMMatoms;i++){
    for(j=0;j<DIM;j++){
      f[i+qm->nrQMatoms][j]      = HARTREE_BOHR2MD*MMgrad[i][j];
      fshift[i+qm->nrQMatoms][j] = HARTREE_BOHR2MD*MMgrad[i][j];
    }
  }
  QMener = QMener*HARTREE2KJ*AVOGADRO;
  fprintf(stderr,"step %5d, SA = %5d, swap = %5d\n",
	  step,(qm->SAstep>0),swapped);
  step++;
  free(exe);
  return(QMener);

} /* call_gaussian_SH */
    
/* end of gaussian sub routines */

#else
int
gmx_qmmm_gaussian_empty;
#endif

