/*=================================================================
  * fd2d_update_i.c -- update Vx, Vz, Txx, Tzz and Txz 
  *
  * 2D finite difference modeling seismic waves propagation in fractured media
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
 *                Vij| dens                    |         X
 *         ----------x-------------------------x---------->  z(k)=(k-1)*dz
 *                   |(m,k)                    |(m+1,k)
 *                   |            .            |
 *                   |            .            |
 *                   |            .            |
 *                   |            .            |
 *                   | . . . . Tij, Cij . . .  |
 *                   |                         |
 *                   |                         |
 *                   |                         |
 *                   |                         |
 *                   |                         |
 *         ----------x-------------------------x----------   z(k+1)=k*dz
 *                   |(m,k+1)                  |(m+1,k+1)
 *                   |                         |
 *                 Z |                         |
 *                   V
 *
 *                x(m)=(m-1)*dx             x(m+1)=m*dx
 *
 *          rotated staggered grid
 *
 *
 *
 *                 m    m+1   m+2   m+3   m+4  m+5      
 *                 |     |     |     |     |     |
 *        k ---    V     V     V     V     V     V   
 *
 *                    T     T     T     T     T     T
 *      k+1 ---    V     V     V     V     V     V   
 *
 *                    T     T     T     T     T     T
 *      k+2 ---    V     V     V     V     V     V   
 *
 *                    T     T     T     T     T     T
 *      k+3 ---    V     V     V     V     V     V   
 *
 *                    T     T     T     T     T     T
 *
 *
 *================================================================================*
 *
 *          d(txx)       d(vx)         d(vz)       ( d(vx)     (dvz) )
 *          ------ = c1  -----  +  c3  ----- + c5  ( -----  +  ----- )
 *            dt           dx           dz         (  dz         dx  )
 *
 *          d(tzz)       d(vx)         d(vz)       ( d(vx)     d(vz) )
 *          ------ = c3  -----  +  c12 ----- + c14 ( -----  +  ----- )
 *            dt           dx           dz         (  dz         dx  )
 *
 *          d(txz)       d(vx)         d(vz)       ( d(vx)     d(vz) )
 *          ------ = c5  -----  +  c14 ----- + c19 ( -----  +  ----- )
 *            dt          dx            dz         (   dz        dx  )
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
void update_T_RSG4(int istep, FILE *fid)
{
  int ik, ipml, GIJ, PIJ, iipro;
  double d1, d2, exx,ezz,exz,ezx, force, Vtmp;
/* for nodes not at top or bottom, don't do two rows near edge */  
  if(ifsbc==1)    ik = 2;  
  else            ik=1;  

  if ( istep * dt <= tsrcf )   force = source_xz(istep,fid);
  if(ifsbc==1)                 update_T_fsbc(istep,fid);	

  if (isourcecomp==3) { /* plane wave */
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
 
/*  update stress Txx, Tzz, Txz at all non-boundary points *****************/
#pragma omp parallel default(shared) private(exx,ezz,exz,ezx,d1,d2,m,k,GIJ,PIJ,iipro,Vtmp,ipml)
{
#pragma omp for
  for(k=ik;k<kk-1;k++) {
    for(m=1;m<mm-1;m++) {
      GIJ=GI(m,k);
      PIJ=PAI(m,k);
      iipro=ipro[m][k];

       if(m==mm-2 || k==kk-2) {
         exx = 0.5*(vx[GI(m+1,k)]-vx[GIJ] + vx[GI(m+1,k+1)]-vx[GI(m,k+1)]);
         ezx = 0.5*(vz[GI(m+1,k)]-vz[GIJ] + vz[GI(m+1,k+1)]-vz[GI(m,k+1)]);
         ezz = 0.5*(vz[GI(m,k+1)]-vz[GIJ] + vz[GI(m+1,k+1)]-vz[GI(m+1,k)]);
         exz = 0.5*(vx[GI(m,k+1)]-vx[GIJ] + vx[GI(m+1,k+1)]-vx[GI(m+1,k)]);
       }
       else {
         d1  = coe1*(vx[GI(m+1,k+1)]-vx[GIJ]) + coe2*(vx[GI(m+2,k+2)]-vx[GI(m-1,k-1)]);
         d2  = coe1*(vx[GI(m+1,k)]-vx[GI(m,k+1)]) + coe2*(vx[GI(m+2,k-1)]-vx[GI(m-1,k+2)]);
         exx = 0.5*(d1+d2);
         exz = 0.5*(d1-d2);
         d1  = coe1*(vz[GI(m+1,k+1)]-vz[GIJ]) + coe2*(vz[GI(m+2,k+2)]-vz[GI(m-1,k-1)]);
         d2  = coe1*(vz[GI(m+1,k)]-vz[GI(m,k+1)]) + coe2*(vz[GI(m+2,k-1)]-vz[GI(m-1,k+2)]);
         ezx = 0.5*(d1+d2);
         ezz = 0.5*(d1-d2);
       }

      if(iipro<0) { // isotropic
      	txx[GIJ] = txx[GIJ] + c[m][k][1]*exx + c[m][k][2]*ezz;
      	tzz[GIJ] = tzz[GIJ] + c[m][k][2]*exx + c[m][k][1]*ezz;
        txz[GIJ] = txz[GIJ] + c[m][k][3]*(exz+ezx);
      }
      else {  // anisotropic
      	txx[GIJ] = txx[GIJ] + cani[iipro][1]*exx + cani[iipro][3]*ezz  + cani[iipro][5]*(exz+ezx);
      	tzz[GIJ] = tzz[GIJ] + cani[iipro][3]*exx + cani[iipro][12]*ezz + cani[iipro][14]*(exz+ezx);
        txz[GIJ] = txz[GIJ] + cani[iipro][5]*exx + cani[iipro][14]*ezz + cani[iipro][19]*(exz+ezx);
      }

      // assume the PML region is isotropic
      // PML at X direction
      if(m<nabs) {
         ipml=nabs-1-m;
         // txx & tzz
         Vtmp = (2.0*pml_vxx[PIJ] + pml_d_half[ipml]*exx)/(2.0 + pml_alpha_half[ipml]);
         txx[GIJ] = txx[GIJ] + c[m][k][1] * ( (pml_beta_half[ipml]-1)*exx - pml_beta_half[ipml]*Vtmp*dx );
         tzz[GIJ] = tzz[GIJ] + c[m][k][2] * ( (pml_beta_half[ipml]-1)*exx - pml_beta_half[ipml]*Vtmp*dx);
         pml_vxx[PIJ] = 2.0*Vtmp - pml_vxx[PIJ];
         // txz
         Vtmp = (2.0*pml_vzx[PIJ] + pml_d_half[ipml]*ezx)/(2.0 + pml_alpha_half[ipml]);
         txz[GIJ] = txz[GIJ] + c[m][k][3] * ( (pml_beta_half[ipml]-1)*ezx - pml_beta_half[ipml]*Vtmp*dx );
         pml_vzx[PIJ] = 2.0*Vtmp - pml_vzx[PIJ];
      }
      if(m>=mm-nabs) {
         ipml=m-(mm-nabs); 
         // txx & tzz 
         Vtmp = (2.0*pml_vxx[PIJ] + pml_d_half[ipml]*exx)/(2.0 + pml_alpha_half[ipml]);
         txx[GIJ] = txx[GIJ] + c[m][k][1] * ( (pml_beta_half[ipml]-1)*exx - pml_beta_half[ipml]*Vtmp*dx );
         tzz[GIJ] = tzz[GIJ] + c[m][k][2] * ( (pml_beta_half[ipml]-1)*exx - pml_beta_half[ipml]*Vtmp*dx);
         pml_vxx[PIJ] = 2.0*Vtmp - pml_vxx[PIJ];
         // txz
         Vtmp = (2.0*pml_vzx[PIJ] + pml_d_half[ipml]*ezx)/(2.0 + pml_alpha_half[ipml]);
         txz[GIJ] = txz[GIJ] + c[m][k][3] * ( (pml_beta_half[ipml]-1)*ezx - pml_beta_half[ipml]*Vtmp*dx );
         pml_vzx[PIJ] = 2.0*Vtmp - pml_vzx[PIJ];

      }

      // PML at Z direction
      if(k<nabs_top) {
         ipml=nabs_top-1-k;
         // txx & tzz
         Vtmp = (2.0*pml_vzz[PIJ] + pml_d_half[ipml]*ezz)/(2.0 + pml_alpha_half[ipml]);
         txx[GIJ] = txx[GIJ] + c[m][k][2] * ( (pml_beta_half[ipml]-1)*ezz-pml_beta_half[ipml]*Vtmp*dx );
         tzz[GIJ] = tzz[GIJ] + c[m][k][1] * ( (pml_beta_half[ipml]-1)*ezz-pml_beta_half[ipml]*Vtmp*dx );
         pml_vzz[PIJ] = 2.0*Vtmp - pml_vzz[PIJ]; 
         // txz
         Vtmp = (2.0*pml_vxz[PIJ] + pml_d_half[ipml]*exz)/(2.0 + pml_alpha_half[ipml]);
         txz[GIJ] = txz[GIJ] + c[m][k][3]  * ( (pml_beta_half[ipml]-1)*exz-pml_beta_half[ipml]*Vtmp*dx );
         pml_vxz[PIJ] = 2.0*Vtmp - pml_vxz[PIJ]; 
      }
      if(k>=kk-nabs) {
         // txz
         ipml=k-(kk-nabs); 
         Vtmp = (2.0*pml_vxz[PIJ] + pml_d_half[ipml]*exz)/(2.0 + pml_alpha_half[ipml]);
         txz[GIJ] = txz[GIJ] + c[m][k][3]  * ( (pml_beta_half[ipml]-1)*exz-pml_beta_half[ipml]*Vtmp*dx );
         pml_vxz[PIJ] = 2.0*Vtmp - pml_vxz[PIJ]; 
         // txx & tzz
         Vtmp = (2.0*pml_vzz[PIJ] + pml_d_half[ipml]*ezz)/(2.0 + pml_alpha_half[ipml]);
         txx[GIJ] = txx[GIJ] + c[m][k][2] * ( (pml_beta_half[ipml]-1)*ezz-pml_beta_half[ipml]*Vtmp*dx );
         tzz[GIJ] = tzz[GIJ] + c[m][k][1] * ( (pml_beta_half[ipml]-1)*ezz-pml_beta_half[ipml]*Vtmp*dx );
         pml_vzz[PIJ] = 2.0*Vtmp - pml_vzz[PIJ]; 
      }
   } }
} // end of parallel

return;
}
/************************************************************/
void update_V_RSG4(int istep, FILE *fid)
{
  int ik, ipml, GIJ, PIJ; 
  double d1, d2, xtxx, ztzz, xtxz, ztxz, Ttmp;

/* for nodes not at top or bottom, don't do two rows near edge */  
  if(ifsbc==1)    ik = 2;  
  else            ik=1;
  
  if(ifsbc==1)    update_V_fsbc(istep,fid);

/* update velocity Vx and Vz at all non-boundary points */
#pragma omp parallel default(shared) private(xtxx,ztzz,ztxz,xtxz,d1,d2,m,k,GIJ,PIJ,Ttmp,ipml)
{
#pragma omp for
  for(k=ik;k<kk-1;k++) {
    for(m=1;m<mm-1;m++) {

       GIJ=GI(m,k);
       PIJ=PAI(m,k);

       if(m==1 || k==1) {
         xtxx = 0.5*(txx[GI(m,k-1)]-txx[GI(m-1,k-1)] + txx[GIJ]-txx[GI(m-1,k)]);
         xtxz = 0.5*(txz[GI(m,k-1)]-txz[GI(m-1,k-1)] + txz[GIJ]-txz[GI(m-1,k)]);
         ztzz = 0.5*(tzz[GI(m-1,k)]-tzz[GI(m-1,k-1)] + tzz[GIJ]-tzz[GI(m,k-1)]);
         ztxz = 0.5*(txz[GI(m-1,k)]-txz[GI(m-1,k-1)] + txz[GIJ]-txz[GI(m,k-1)]);
       }
       else {
         d1 = coe1*(txx[GIJ]-txx[GI(m-1,k-1)]) + coe2*(txx[GI(m+1,k+1)]-txx[GI(m-2,k-2)]);
         d2 = coe1*(txx[GI(m,k-1)]-txx[GI(m-1,k)]) + coe2*(txx[GI(m+1,k-2)]-txx[GI(m-2,k+1)]);
         xtxx = 0.5*(d1+d2);
         d1 = coe1*(tzz[GIJ]-tzz[GI(m-1,k-1)]) + coe2*(tzz[GI(m+1,k+1)]-tzz[GI(m-2,k-2)]);
         d2 = coe1*(tzz[GI(m,k-1)]-tzz[GI(m-1,k)]) + coe2*(tzz[GI(m+1,k-2)]-tzz[GI(m-2,k+1)]);
         ztzz = 0.5*(d1-d2);
         d1 = coe1*(txz[GIJ]-txz[GI(m-1,k-1)]) + coe2*(txz[GI(m+1,k+1)]-txz[GI(m-2,k-2)]);
         d2 = coe1*(txz[GI(m,k-1)]-txz[GI(m-1,k)]) + coe2*(txz[GI(m+1,k-2)]-txz[GI(m-2,k+1)]);
         xtxz = 0.5*(d1+d2);
         ztxz = 0.5*(d1-d2);
       }
       // dens'=dt/dx*1/dens
       vx[GIJ] = vx[GIJ] + (xtxx+ztxz)*c[m][k][0];
       vz[GIJ] = vz[GIJ] + (xtxz+ztzz)*c[m][k][0];

      // PML at X direction
      if(m<nabs) {
         ipml=nabs-1-m;
         // vx 
         Ttmp = (2.0*pml_xtxx[PIJ] + pml_d[ipml]*xtxx)/(2.0+pml_alpha[ipml]);
         vx[GIJ] = vx[GIJ] + ((pml_beta[ipml]-1)*xtxx - pml_beta[ipml]*Ttmp*dx)*c[m][k][0];
         pml_xtxx[PIJ] = 2.0*Ttmp - pml_xtxx[PIJ];
         // vz
         Ttmp = (2.0*pml_xtxz[PIJ] + pml_d[ipml]*xtxz)/(2.0+pml_alpha[ipml]);
         vz[GIJ] = vz[GIJ] + ((pml_beta[ipml]-1)*xtxz - pml_beta[ipml]*Ttmp*dx)*c[m][k][0];
         pml_xtxz[PIJ] = 2.0*Ttmp - pml_xtxz[PIJ];
      }
      if(m>mm-nabs) {
         // vz
         ipml=m-(mm-nabs)-1; 
         Ttmp = (2.0*pml_xtxz[PIJ] + pml_d[ipml]*xtxz)/(2.0+pml_alpha[ipml]);
         vz[GIJ] = vz[GIJ] + ((pml_beta[ipml]-1)*xtxz - pml_beta[ipml]*Ttmp*dx)*c[m][k][0];
         pml_xtxz[PIJ] = 2.0*Ttmp - pml_xtxz[PIJ];
         // vx 
         Ttmp = (2.0*pml_xtxx[PIJ] + pml_d[ipml]*xtxx)/(2.0+pml_alpha[ipml]);
         vx[GIJ] = vx[GIJ] + ((pml_beta[ipml]-1)*xtxx - pml_beta[ipml]*Ttmp*dx)*c[m][k][0];
         pml_xtxx[PIJ] = 2.0*Ttmp - pml_xtxx[PIJ];
      }

      // PML at Z direction
      if(k<nabs_top) {
         ipml=nabs_top-1-k;
         // vx
         Ttmp = (2.0*pml_ztxz[PIJ] + pml_d[ipml]*ztxz)/(2.0+pml_alpha[ipml]);
         vx[GIJ] = vx[GIJ] + ((pml_beta[ipml]-1)*ztxz - pml_beta[ipml]*Ttmp*dx)*c[m][k][0];
         pml_ztxz[PIJ] = 2.0*Ttmp - pml_ztxz[PIJ];
         // vz
         Ttmp = (2.0*pml_ztzz[PIJ] + pml_d[ipml]*ztzz)/(2.0+pml_alpha[ipml]);
         vz[GIJ] = vz[GIJ] + ((pml_beta[ipml]-1)*ztzz - pml_beta[ipml]*Ttmp*dx)*c[m][k][0];
         pml_ztzz[PIJ] = 2.0*Ttmp - pml_ztzz[PIJ];
      }
      if(k>kk-nabs) {
         // vz
         ipml=k-(kk-nabs)-1; 
         Ttmp = (2.0*pml_ztzz[PIJ] + pml_d[ipml]*ztzz)/(2.0+pml_alpha[ipml]);
         vz[GIJ] = vz[GIJ] + ((pml_beta[ipml]-1)*ztzz - pml_beta[ipml]*Ttmp*dx)*c[m][k][0];
         pml_ztzz[PIJ] = 2.0*Ttmp - pml_ztzz[PIJ];
         // vx
         Ttmp = (2.0*pml_ztxz[PIJ] + pml_d[ipml]*ztxz)/(2.0+pml_alpha[ipml]);
         vx[GIJ] = vx[GIJ] + ((pml_beta[ipml]-1)*ztxz - pml_beta[ipml]*Ttmp*dx)*c[m][k][0];
         pml_ztxz[PIJ] = 2.0*Ttmp - pml_ztxz[PIJ];
      }
  } }
} // end of parallel

 return;
}


