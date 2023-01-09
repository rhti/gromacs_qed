/*
 * 
 *                This source code is part of
 * 
 *                 G   R   O   M   A   C   S
 * 
 *          GROningen MAchine for Chemical Simulations
 * 
 * Written by David van der Spoel, Erik Lindahl, Berk Hess, and others.
 * Copyright (c) 1991-2000, University of Groningen, The Netherlands.
 * Copyright (c) 2001-2008, The GROMACS development team,
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
 * Gallium Rubidium Oxygen Manganese Argon Carbon Silicon
 */


#include "simple.h"
#include "complex.h"
#ifdef I
#undef I
#endif
#define IMAG _Complex_I
typedef double complex dplx;
#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
 int           nrQMatoms;      /* total nr of QM atoms              */
 rvec          *xQM;           /* shifted to center of box          */  
 int           *indexQM;       /* atom i = atom indexQM[i] in mdrun */
 int           *atomicnumberQM;/* atomic numbers of QM atoms        */  
 real          *QMcharges;     /* atomic charges of QM atoms(ONIOM) */
 int           *shiftQM;
 int           QMcharge;       /* charge of the QM system           */
 int           multiplicity;   /* multipicity (no of unpaired eln)  */
 int           QMmethod;       /* see enums.h for all methods       */
 int           QMbasis;        /* see enums.h for all bases         */
 int           nelectrons;     /* total number of elecs in QM region*/
 gmx_bool          bTS;            /* Optimize a TS, only steep, no md  */
 gmx_bool          bOPT;          /* Optimize QM subsys, only steep, no md  */
 gmx_bool          *frontatoms;   /* qm atoms on the QM side of a QM-MM bond */
 /* Gaussian specific stuff */
 int           nQMcpus;        /* no. of CPUs used for the QM calc. */
 int           QMmem;          /* memory for the gaussian calc.     */
 int           accuracy;       /* convergence criterium (E(-x))     */
 gmx_bool          cpmcscf;        /* using cpmcscf(l1003)*/
 char          *gauss_dir;
 char          *gauss_exe;
 char          *devel_dir;
 char          *orca_basename; /* basename for I/O with orca        */
 char          *orca_dir;      /* directory for ORCA                */
 real          *c6;
 real          *c12;
 /* Surface hopping stuff */
 gmx_bool          bSH;            /* surface hopping (diabatic only)   */
 real          SAon;           /* at which energy gap the SA starts */
 real          SAoff;          /* at which energy gap the SA stops  */
 int           SAsteps;        /* stepwise switchinng on the SA     */
 int           SAstep;         /* current state of SA               */
 int           CIdim;
 real          *CIvec1;
 real          *CIvec2;
 real          *CIvec1old;
 real          *CIvec2old;
 ivec          SHbasis;
 int           CASelectrons;
 int           CASorbitals;
  /* for cavity QED/MM */
  char *subdir;
  int polariton;
  double *creal; /* expansion coefficeint vector */
  double *cimag;
double groundstate;
  double QEDdecay;
  
  double dt;/*timestep from inputrec */
  real omega ;   /* energy of the cavity EM mode */
  rvec E;
  real    n_index; /* Refractive index */
  real    L; /* Cavity length */
  int    n_min; /* integer defining the minimum k=2*Pi*n_min/L_au */
  int    n_max; /* integer defining the maximum k=2*Pi*n_max/L_au */
  double *z; /* Molecules position along the z axis vector */

  dplx  *eigvec; /* to keep previous vectors, needed for hopping
		  */
  double *eigval; /* to keep prevous energies, needed for hopping */
  gmx_bool bQED; /* cavity QED with multiple molecules */
  gmx_bool bContinuation;   /* Continuation run: copied here to link it with QEDrestart (ideally both are True or False) */
  gmx_bool QEDrestart; /* for restarting cavity QED with well defined eigenvectors phases */
  rvec     *xref;
  rvec     *tdmXgrad;
  rvec     *tdmYgrad;
  rvec     *tdmZgrad;
  rvec     tdm;
  rvec     tdmold;

  char *work_dir; /* where all important input gets stored */
  real     *rnr; /* random number string... */
  int      nsteps;
  double     QEDdecoherence; /* value of decoherence parameter. If > 0, the do something */
  rvec *vQM; /* qm velocities */
  real *ffmass; /* qm atoms masses */
} t_QMrec;

typedef struct {
  int           nrMMatoms;      /* nr of MM atoms, updated every step*/
  rvec          *xMM;           /* shifted to center of box          */
  int           *indexMM;       /* atom i = atom indexMM[I] in mdrun */
  real          *MMcharges;     /* MM point charges in std QMMM calc.*/
  int           *shiftMM;
  int           *MMatomtype;    /* only important for semi-emp.      */
  real          scalefactor;
  rvec *vMM;
  real *ffmass;
  /* gaussian specific stuff */
  real          *c6;
  real          *c12;
} t_MMrec;


typedef struct {
  int           QMMMscheme; /* ONIOM (multi-layer) or normal          */
  int           nrQMlayers; /* number of QM layers (total layers +1 (MM)) */
  t_QMrec       **qm;        /* atoms and run params for each QM group */
  t_MMrec       *mm;        /* there can only be one MM subsystem !   */
  int           SHmethod;
} t_QMMMrec;

#ifdef __cplusplus
}
#endif
