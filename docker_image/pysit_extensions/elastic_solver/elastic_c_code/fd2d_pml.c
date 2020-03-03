/*==========================================================================
  * fd2d_pml.c -- setup PML damping values
  *        
  * 2D finite difference modeling seismic waves propagation in fractured media
  * 
  *-----------------------------------------------------------------------------------------------
  * Log:
       2010-9-28: implement Wei Zhang's unsplit complex frequency-shifted PML, Xinding & Xuefeng
  *-----------------------------------------------------------------------------------------------
  *
  *  Xinding Fang, MIT-ERL
  *  Email: xinfang@mit.edu
  *
  * Reference:
  * (1) Zhang, W. and Y. Shen, 2010. Unsplit complex frequency-shifted PML implementation 
  *     using auxiliary differential equations for seimsic wave modeling: Geophysics, 75, 141-154.
  * (2) Marcinkovich C. and K. Olsen, 2003. On the implementation of perfectly matched layers in 
  *     a three-dimensional fourth-order velocity-stress finite difference scheme: 
  *     Journal of Geophysical Research, 108, NO. B5, 2276
 ===========================================================================*/
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "fd2d.h"
#include "funlist.h"

void setup_damping(FILE *fid)
{
  double lnR, thk_pml, beta0, alpha0, d0, xL;	

  // natural logarithm of the theoretical reflection coefficient R
  lnR=log(10)*(-3.0-(log10(nabs+0.0)-1.0)/log10(2.0)); 
  // thickness of PML
  thk_pml=nabs*dx;

  d0=-(p_power+1.0)*Vmax*lnR/(2.0*thk_pml);
  d0=d0*d0factor;
  beta0=Vmax/(0.5*PPW0*dx*freq0);
  if(beta0<1.0)  beta0=1.0;
  alpha0=pi*freq0;

  if(ifsbc==0)  fprintf(fid,"    No free surface \n");
  else          fprintf(fid,"    Free surface on top of the model \n");
  fprintf(fid,"    PML Parameters: \n");
  fprintf(fid,"        thickness (grid): %i \n",nabs);
  fprintf(fid,"        p_power=%g d0factor=%g PPW0=%g Vmax=%g (m/s) freq0=%g (Hz)\n",p_power,d0factor,PPW0,Vmax,freq0);
  fprintf(fid,"        d0=%g beta0=%g alpha0=%g \n",d0,beta0,alpha0);

  for(i=0;i<nabs;i++) {
      // i=0: interior interface; i=nabs-1: exterior boundry; 

      // define damping profile at grid points
      xL=(i+1)*dx/thk_pml;
      pml_d[i]=d0*pow(xL,p_power);
      pml_beta[i]=1.0+(beta0-1.0)*pow(xL,p_power);
      pml_alpha[i]=alpha0*(1.0-xL);

      // define damping profile at half grid points
      xL=(i+0.5)*dx/thk_pml;
      pml_d_half[i]=d0*pow(xL,p_power);
      pml_beta_half[i]=1.0+(beta0-1.0)*pow(xL,p_power);
      pml_alpha_half[i]=alpha0*(1.0-xL);

      if(pml_alpha[i]<0.0)       pml_alpha[i]=0.0;
      if(pml_alpha_half[i]<0.0)  pml_alpha_half[i]=0.0;

      // beta <-- 1/beta
      pml_beta[i]=1.0/pml_beta[i];
      pml_beta_half[i]=1.0/pml_beta_half[i];

      // d <-- d/beta 
      pml_d[i]=pml_d[i]*pml_beta[i];
      pml_d_half[i]=pml_d_half[i]*pml_beta_half[i];

      // alpha <-- alpha + d/beta 
      pml_alpha[i]=pml_alpha[i]+pml_d[i];
      pml_alpha_half[i]=pml_alpha_half[i]+pml_d_half[i];
/*
      // multiply d/beta and alpha+d/beta by dt
      pml_d[i]=dt*pml_d[i];
      pml_d_half[i]=dt*pml_d_half[i];
      pml_alpha[i]=dt*pml_alpha[i];
      pml_alpha_half[i]=dt*pml_alpha_half[i];
*/
      // multiply alpha+d/beta by dt
      pml_alpha[i]=dt*pml_alpha[i];
      pml_alpha_half[i]=dt*pml_alpha_half[i];
      // multiply d/beta by dt/dx
      pml_d[i]=dt/dx*pml_d[i];
      pml_d_half[i]=dt/dx*pml_d_half[i];
  }
/*
    for(i=0;i<nabs;i++) {
        fprintf(fid,"%i: beta=%g d=%g alpha=%g \n",i,pml_beta[i],pml_d[i],pml_alpha[i]);
        fprintf(fid,"    beta=%g d=%g alpha=%g \n",pml_beta_half[i],pml_d_half[i],pml_alpha_half[i]);
    }
*/
return;
} /* end */


