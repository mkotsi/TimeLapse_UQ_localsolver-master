/*=================================================================
  * fd2d_update_SSG_Acoustic.c -- solving acoustic wave equation by implementing SSG
  * 
  *  Xinding Fang, MIT-ERL, May 2009
  *  Email: xinfang@mit.edu
 =================================================================*/
 
/***********************************************************************
 *   4th-order in space and 2nd-order in time
 *   Time Domain Finite Difference in 2-D Cartesian Coord's
 *
 *	this subroutine updates velocity Vx,Vz
 *		                  stress Txx,Tzz,Txz
 *       on cartesian coordinate (x,z) grid staggered in space & time:
 *                               stresses   @ t = k dt, ...
 *                               velocities @ t = (k+1/2) dt, ...
 *
 *
 *                   |                         |
 *                   |                         |
 *                Vx |       Tii,dens,Cij      |         X
 *         ----------x------------o------------x---------->  z(k)=(k-1)*dz
 *                   |(m,k)       (m+1/2,k)    |(m+1,k)
 *                   |            .            |
 *                   |            .            |
 *                   |            .            |
 *                   |            .            |
 *                   |          Vz.            |
 *                   o . . . . . .o            o
 *                   |(m,k+1/2)   (m+1/2,k+1/2)|(m+1,k+1/2)
 *                   |                         |
 *                   |                         |
 *                   |                         |
 *                   |                         |
 *                   |                         |
 *         ----------x------------o------------x----------   z(k+1)=k*dz
 *                   |(m,k+1)     (m+1/2,k+1)  |(m+1,k+1)
 *                   |                         |
 *                 Z |                         |
 *                   V
 *
 *                x(m)=(m-1)*dx             x(m+1)=m*dx
 *
 *
 *
 *                 m             m+1            m+2         
 *                 |              |              |
 *        k ---    Vx     Tii     Vx     Tii     Vx     Tii     
 *
 *                        Vz             Vz             Vz     
 *
 *      k+1 ---    Vx     Tii     Vx     Tii     Vx     Tii    
 *
 *                        Vz             Vz             Vz   
 *
 *      k+2 ---    Vx     Tii     Vx     Tii     Vx     Tii    
 *
 *                        Vz             Vz             Vz    
 *
 *
 *================================================================================*
 *
 *          d(txx)        ( d(vx)      d(vz) )    
 *          ------ = c1 * ( -----  +   ----- )
 *            dt          (  dx         dz   )
 *
 *          tzz=txx, txz=0.   
 *
 *
 ********************************************************************/
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "fd2d.h"
#include "funlist.h"

/*************************************************************/
/*************************************************************/
void update_T_SSG_Acoustic(int istep, FILE *fid)
{
  int ik, ipml;
  double exx,ezz, force, Vtmp;
  unsigned long GIJ, PIJ;

  if(ifsbc==1)    ik = 2;  
  else            ik = 1;  

  if (istep * dt <= tsrcf)   force = source_xz(istep,fid);

  if(ifsbc==1)               update_T_fsbc(istep,fid);	

  if (isourcecomp==3) { /* plane wave */
    if(isdir==0) { // x-direction
      for(m=0;m<mm;m++) {
        for(k=5;k>=0;k--) {
          vx[GI(m,k)]=vx[GI(m,kk-11+k)]; 
          vz[GI(m,k)]=vz[GI(m,kk-11+k)];
        }
        for(k=kk-5;k<kk;k++) {
          vx[GI(m,k)]=vx[GI(m,k-kk+11)]; 
          vz[GI(m,k)]=vz[GI(m,k-kk+11)]; 
        }
      }
    }
    else {  // z-direction
      for(k=0;k<kk;k++) {
         for(m=5;m>=0;m--) {
            vx[GI(m,k)]=vx[GI(mm-11+m,k)];
            vz[GI(m,k)]=vz[GI(mm-11+m,k)];
         }
         for(m=mm-5;m<mm;m++) {
            vx[GI(m,k)]=vx[GI(m-mm+11,k)];
            vz[GI(m,k)]=vz[GI(m-mm+11,k)];
         }
      }
    }
  }

/*  update stress Txx, Tzz, Txz at all non-boundary points *****************/

#pragma omp parallel default(shared) private(exx,ezz,m,k,GIJ,PIJ,Vtmp,ipml)
{
#pragma omp for
  for(k=ik;k<kk-1;k++) {
    for(m=1;m<mm-1;m++) {
       GIJ=GI(m,k);

       if(m==1 || m==mm-2)  exx = vx[GI(m+1,k)] - vx[GIJ];
       else                 exx = coe1*(vx[GI(m+1,k)]-vx[GIJ]) + coe2*(vx[GI(m+2,k)]-vx[GI(m-1,k)]); 

       if(k==1 || k==kk-2)  ezz = vz[GIJ] - vz[GI(m,k-1)];
       else                 ezz = coe1*(vz[GIJ]-vz[GI(m,k-1)]) + coe2*(vz[GI(m,k+1)]-vz[GI(m,k-2)]);  

       txx[GIJ] = txx[GIJ] + c[m][k][1] * (exx + ezz);

      // assume the PML region is isotropic
      // PML at X direction
      if(m<nabs) {
         PIJ=PAI(m,k);
         ipml=nabs-1-m;
         Vtmp = (2.0*pml_vxx[PIJ] + pml_d_half[ipml]*exx)/(2.0 + pml_alpha_half[ipml]);
         txx[GIJ] = txx[GIJ] + c[m][k][1] * ( (pml_beta_half[ipml]-1)*exx - pml_beta_half[ipml]*Vtmp*dx );
         pml_vxx[PIJ] = 2.0*Vtmp - pml_vxx[PIJ];
      }
      if(m>=mm-nabs) {
         PIJ=PAI(m,k);
         ipml=m-(mm-nabs); 
         Vtmp = (2.0*pml_vxx[PIJ] + pml_d_half[ipml]*exx)/(2.0 + pml_alpha_half[ipml]);
         txx[GIJ] = txx[GIJ] + c[m][k][1] * ( (pml_beta_half[ipml]-1)*exx - pml_beta_half[ipml]*Vtmp*dx);
         pml_vxx[PIJ] = 2.0*Vtmp - pml_vxx[PIJ];
      }
      // PML at Z direction
      if(k<nabs_top) {
         PIJ=PAI(m,k);
         ipml=nabs_top-1-k;
         Vtmp = (2.0*pml_vzz[PIJ] + pml_d[ipml]*ezz)/(2.0 + pml_alpha[ipml]);
         txx[GIJ] = txx[GIJ] + c[m][k][1] * ( (pml_beta[ipml]-1)*ezz-pml_beta[ipml]*Vtmp*dx );
         pml_vzz[PIJ] = 2.0*Vtmp - pml_vzz[PIJ]; 
      }
      if(k>kk-nabs) {
         PIJ=PAI(m,k);
         ipml=k-(kk-nabs)-1; 
         Vtmp = (2.0*pml_vzz[PIJ] + pml_d[ipml]*ezz)/(2.0 + pml_alpha[ipml]);
         txx[GIJ] = txx[GIJ] + c[m][k][1] * ( (pml_beta[ipml]-1)*ezz-pml_beta[ipml]*Vtmp*dx );
         pml_vzz[PIJ] = 2.0*Vtmp - pml_vzz[PIJ]; 
      }

      tzz[GIJ] = txx[GIJ];
   } }
} // end of pragma omp parallel
return;
}

/************************************************************/
void update_V_SSG_Acoustic(int istep, FILE *fid)
{
  int ik, ipml;
  double xtxx, ztzz, Ttmp;
  unsigned long GIJ, PIJ;

/* for nodes not at top or bottom, don't do two rows near edge */  
  if(ifsbc==1)    ik = 2;  
  else            ik = 1;
  
  if(ifsbc==1)    update_V_fsbc(istep,fid);

/* update velocity Vx and Vz at all non-boundary points */
#pragma omp parallel default(shared) private(xtxx,ztzz,m,k,GIJ,PIJ,Ttmp,ipml)
{
#pragma omp for
  for(k=ik;k<kk-1;k++) {
    for(m=1;m<mm-1;m++) {
      GIJ=GI(m,k);

      if(m==1 || m==mm-2)  xtxx = txx[GIJ]-txx[GI(m-1,k)];
      else 	           xtxx = coe1*(txx[GIJ]  -txx[GI(m-1,k)]) + coe2*(txx[GI(m+1,k)]-txx[GI(m-2,k)]);
      
      if(k==1 || k==kk-2)  ztzz = tzz[GI(m,k+1)]-tzz[GIJ];
      else                 ztzz = coe1*(tzz[GI(m,k+1)]-tzz[GIJ]) + coe2*(tzz[GI(m,k+2)]-tzz[GI(m,k-1)]);

      // dtdx = dt/dx 
      vx[GIJ] = vx[GIJ] + xtxx*2.0*dtdx/(c[m][k][0]+c[m-1][k][0]);
      vz[GIJ] = vz[GIJ] + ztzz*2.0*dtdx/(c[m][k][0]+c[m][k+1][0]);

      // PML at X direction
      if(m<nabs) {
         PIJ=PAI(m,k);
         ipml=nabs-1-m;
         Ttmp = (2.0*pml_xtxx[PIJ] + pml_d[ipml]*xtxx)/(2.0+pml_alpha[ipml]);
         vx[GIJ] = vx[GIJ] + ((pml_beta[ipml]-1)*xtxx/dx - pml_beta[ipml]*Ttmp)*dt*2.0/(c[m][k][0]+c[m-1][k][0]);
         pml_xtxx[PIJ] = 2.0*Ttmp - pml_xtxx[PIJ];
      }
      if(m>mm-nabs) {
         PIJ=PAI(m,k);
         ipml=m-(mm-nabs)-1; 
         Ttmp = (2.0*pml_xtxx[PIJ] + pml_d[ipml]*xtxx)/(2.0+pml_alpha[ipml]);
         vx[GIJ] = vx[GIJ] + ((pml_beta[ipml]-1)*xtxx/dx - pml_beta[ipml]*Ttmp)*dt*2.0/(c[m][k][0]+c[m-1][k][0]);
         pml_xtxx[PIJ] = 2.0*Ttmp - pml_xtxx[PIJ];
      }

      // PML at Z direction
      if(k<nabs_top) {
         PIJ=PAI(m,k);
         ipml=nabs_top-1-k;
         Ttmp = (2.0*pml_ztzz[PIJ] + pml_d_half[ipml]*ztzz)/(2.0+pml_alpha_half[ipml]);
         vz[GIJ] = vz[GIJ] + ((pml_beta_half[ipml]-1)*ztzz/dx - pml_beta_half[ipml]*Ttmp)*dt*2.0/(c[m][k][0]+c[m-1][k][0]);
         pml_ztzz[PIJ] = 2.0*Ttmp - pml_ztzz[PIJ];
      }
      if(k>=kk-nabs) {
         PIJ=PAI(m,k);
         ipml=k-(kk-nabs); 
         Ttmp = (2.0*pml_ztzz[PIJ] + pml_d_half[ipml]*ztzz)/(2.0+pml_alpha_half[ipml]);
         vz[GIJ] = vz[GIJ] + ((pml_beta_half[ipml]-1)*ztzz/dx - pml_beta_half[ipml]*Ttmp)*dt*2.0/(c[m][k][0]+c[m-1][k][0]);
         pml_ztzz[PIJ] = 2.0*Ttmp - pml_ztzz[PIJ];
      }
  } }
} // end of pragma omp parallel
 return;
}


