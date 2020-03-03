/*=================================================================
  * fd2d_update_i_SSG.c -- update Vx, Vz, Txx, Tzz and Txz 
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
 *                Vx |       Tii,dens,Cij      |         X
 *         ----------x------------o------------x---------->  z(k)=(k-1)*dz
 *                   |(m,k)       (m+1/2,k)    |(m+1,k)
 *                   |            .            |
 *                   |            .            |
 *                   |            .            |
 *                   |            .            |
 *               Txz |          Vz.            |
 *               Axz o . . . . . .o            o
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
 *                 Txz    Vz      Txz    Vz      Txz    Vz     
 *
 *      k+1 ---    Vx     Tii     Vx     Tii     Vx     Tii    
 *
 *                 Txz    Vz      Txz    Vz      Txz    Vz   
 *
 *      k+2 ---    Vx     Tii     Vx     Tii     Vx     Tii    
 *
 *                 Txz    Vz      Txz    Vz      Txz    Vz    
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
 *************************************************************************/
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "fd2d.h"
#include "funlist.h"

/*************************************************************/
/*************************************************************/
void update_T_SSG(int istep, FILE *fid,double* boundary_wavefields)
{
  int ik, ipml, iipro;
  double exx,ezz,exz,ezx,c5, c14, c19, force, Vtmp, tmp;
  double exx1, ezz1, exz1, ezx1;       
  unsigned long GIJ, PIJ;
  /* exx,  ezz,  exz  and ezx  --- Txx, Tzz
     exx1, ezz1, exz1 and ezx1 --- Txz      */

  if(ifsbc==1)    ik = 2;  
  else            ik = 1;  

  if ( istep * dt <= tsrcf )   force = source_xz(istep,fid);
  if(ifsbc==1)                 update_T_fsbc(istep,fid);	
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
  get_Dv(fid);
#pragma omp parallel default(shared) private(c19,exx,ezz,exz,ezx,exx1,ezz1,exz1,ezx1,m,k,GIJ,PIJ,iipro,Vtmp,tmp,ipml)
{
#pragma omp for
  for(k=ik;k<kk-1;k++) {
    for(m=1;m<mm-1;m++) {
      GIJ=GI(m,k);
      PIJ=PAI(m,k);
      iipro=ipro[m][k];

      exx = Dvxdx[GIJ];
      ezz = Dvzdz[GIJ];
      exz = Dvxdz[GIJ];
      ezx = Dvzdx[GIJ];

      if(iipro<0) { // isotropic
         c19 = averag(c[m][k][3],c[m-1][k][3],c[m][k+1][3],c[m-1][k+1][3]);

         txx[GIJ] = txx[GIJ] + c[m][k][1]*exx + c[m][k][2]*ezz;
         tzz[GIJ] = tzz[GIJ] + c[m][k][2]*exx + c[m][k][1]*ezz;
         txz[GIJ] = txz[GIJ] + c19*(exz+ezx);

         /*if (istep%100 == 0 &&(exx>1.e-16 || exx<-1.e-16)){
        	 printf("z ind: %i, x ind: %i \n\n", k-nabs, m-nabs);
        	 printf("vxx %e vs vzz %e vs e_t %e\n",exx, ezz, 0.5*(exz+ezx));
        	 printf("txx %e vs tzz %e vs txz %e\n",txx[GIJ], tzz[GIJ], txz[GIJ]);
         	 printf("c[m][k][0]: %e, dens         : %e \n",c[m][k][0],c[m][k][0]);
         	 printf("c[m][k][1]: %e, lambda + 2 mu: %e \n",c[m][k][1],c[m][k][1]/dtdx);
         	 printf("c[m][k][2]: %e, lambda       : %e \n",c[m][k][2],c[m][k][2]/dtdx);
         	 printf("c[m][k][3]: %e, mu           : %e \n",c[m][k][3],c[m][k][3]/dtdx);
         }*/
      }
      else { // anisotropic
         //c5  = averag(c[m][k][5], c[m-1][k][5], c[m][k+1][5], c[m-1][k+1][5]);
         //c14 = averag(c[m][k][14],c[m-1][k][14],c[m][k+1][14],c[m-1][k+1][14]);
         //c19 = averag(c[m][k][19],c[m-1][k][19],c[m][k+1][19],c[m-1][k+1][19]);

         exx1 = 0.25 * (exx+Dvxdx[GI(m-1,k)]+Dvxdx[GI(m-1,k+1)]+Dvxdx[GI(m,k+1)]);
         ezz1 = 0.25 * (ezz+Dvzdz[GI(m-1,k)]+Dvzdz[GI(m-1,k+1)]+Dvzdz[GI(m,k+1)]);
         exz1 = 0.25 * (exz+Dvxdz[GI(m,k-1)]+Dvxdz[GI(m+1,k-1)]+Dvxdz[GI(m+1,k)]);
         ezx1 = 0.25 * (ezx+Dvzdx[GI(m,k-1)]+Dvzdx[GI(m+1,k-1)]+Dvzdx[GI(m+1,k)]);

         txx[GIJ] = txx[GIJ] + cani[iipro][1]*exx  + cani[iipro][3]*ezz   + cani[iipro][5]*(exz1+ezx1);
         tzz[GIJ] = tzz[GIJ] + cani[iipro][3]*exx  + cani[iipro][12]*ezz  + cani[iipro][14]*(exz1+ezx1);
         txz[GIJ] = txz[GIJ] + cani[iipro][5]*exx1 + cani[iipro][14]*ezz1 + cani[iipro][19]*(exz+ezx);
      }
      // assume the PML region is isotropic
      // PML at X direction
      if(m<nabs) {
         ipml=nabs-1-m;
         // txx & tzz
         Vtmp = (2.0*pml_vxx[PIJ] + pml_d_half[ipml]*exx)/(2.0 + pml_alpha_half[ipml]);
         tmp=(pml_beta_half[ipml]-1)*exx - pml_beta_half[ipml]*Vtmp*dx;
         txx[GIJ] = txx[GIJ] + c[m][k][1] * tmp;
         tzz[GIJ] = tzz[GIJ] + c[m][k][2] * tmp;
         pml_vxx[PIJ] = 2.0*Vtmp - pml_vxx[PIJ];
         // txz
         Vtmp = (2.0*pml_vzx[PIJ] + pml_d[ipml]*ezx)/(2.0 + pml_alpha[ipml]);
         txz[GIJ] = txz[GIJ] + c19 * ( (pml_beta[ipml]-1)*ezx - pml_beta[ipml]*Vtmp*dx );
         pml_vzx[PIJ] = 2.0*Vtmp - pml_vzx[PIJ];
      }
      if(m>=mm-nabs) {
         ipml=m-(mm-nabs); 
         // txx & tzz 
         Vtmp = (2.0*pml_vxx[PIJ] + pml_d_half[ipml]*exx)/(2.0 + pml_alpha_half[ipml]);
         txx[GIJ] = txx[GIJ] + c[m][k][1] * ( (pml_beta_half[ipml]-1)*exx - pml_beta_half[ipml]*Vtmp*dx );
         tzz[GIJ] = tzz[GIJ] + c[m][k][2] * ( (pml_beta_half[ipml]-1)*exx - pml_beta_half[ipml]*Vtmp*dx);
         pml_vxx[PIJ] = 2.0*Vtmp - pml_vxx[PIJ];
      }
      if(m>mm-nabs) {
         // txz
         ipml=m-(mm-nabs)-1;
         Vtmp = (2.0*pml_vzx[PIJ] + pml_d[ipml]*ezx)/(2.0 + pml_alpha[ipml]);
         txz[GIJ] = txz[GIJ] + c19 * ( (pml_beta[ipml]-1)*ezx - pml_beta[ipml]*Vtmp*dx );
         pml_vzx[PIJ] = 2.0*Vtmp - pml_vzx[PIJ];
      }
      // PML at Z direction
      if(k<nabs_top) {
         ipml=nabs_top-1-k;
         // txx & tzz
         Vtmp = (2.0*pml_vzz[PIJ] + pml_d[ipml]*ezz)/(2.0 + pml_alpha[ipml]);
         txx[GIJ] = txx[GIJ] + c[m][k][2] * ( (pml_beta[ipml]-1)*ezz-pml_beta[ipml]*Vtmp*dx );
         tzz[GIJ] = tzz[GIJ] + c[m][k][1] * ( (pml_beta[ipml]-1)*ezz-pml_beta[ipml]*Vtmp*dx );
         pml_vzz[PIJ] = 2.0*Vtmp - pml_vzz[PIJ]; 
         // txz
         Vtmp = (2.0*pml_vxz[PIJ] + pml_d_half[ipml]*exz)/(2.0 + pml_alpha_half[ipml]);
         txz[GIJ] = txz[GIJ] + c19  * ( (pml_beta_half[ipml]-1)*exz-pml_beta_half[ipml]*Vtmp*dx );
         pml_vxz[PIJ] = 2.0*Vtmp - pml_vxz[PIJ]; 
      }
      if(k>=kk-nabs) {
         // txz
         ipml=k-(kk-nabs); 
         Vtmp = (2.0*pml_vxz[PIJ] + pml_d_half[ipml]*exz)/(2.0 + pml_alpha_half[ipml]);
         txz[GIJ] = txz[GIJ] + c19  * ( (pml_beta_half[ipml]-1)*exz-pml_beta_half[ipml]*Vtmp*dx );
         pml_vxz[PIJ] = 2.0*Vtmp - pml_vxz[PIJ]; 
      }
      if(k>kk-nabs) {
         // txx & tzz
         ipml=k-(kk-nabs)-1;
         Vtmp = (2.0*pml_vzz[PIJ] + pml_d[ipml]*ezz)/(2.0 + pml_alpha[ipml]);
         txx[GIJ] = txx[GIJ] + c[m][k][2] * ( (pml_beta[ipml]-1)*ezz-pml_beta[ipml]*Vtmp*dx );
         tzz[GIJ] = tzz[GIJ] + c[m][k][1] * ( (pml_beta[ipml]-1)*ezz-pml_beta[ipml]*Vtmp*dx );
         pml_vzz[PIJ] = 2.0*Vtmp - pml_vzz[PIJ]; 
      }
   } }
} // end of pragma omp parallel
if (bparams.local_solve){ //If we are performing a local solve, update the stress around the boundary cells
	if (jj > 0){ //Stress is always updated first, before velocity. At timestep 0 there is no previous recorded velocity to correct the stress with.
		update_boundary_stress_SSG(boundary_wavefields);
	}
}
return;
}

/************************************************************/
void update_V_SSG(int istep, FILE *fid, double* boundary_wavefields)
{
  int ik, ipml;
  double xtxx, ztzz, xtxz, ztxz, Ttmp;
  unsigned long GIJ, PIJ;
/* for nodes not at top or bottom, don't do two rows near edge */  
  if(ifsbc==1)    ik = 2;  
  else            ik = 1;
  
  if(ifsbc==1)    update_V_fsbc(istep,fid);

/* update velocity Vx and Vz at all non-boundary points */
#pragma omp parallel default(shared) private(xtxx,ztzz,ztxz,xtxz,m,k,GIJ,PIJ,Ttmp,ipml)
{
#pragma omp for
  for(k=ik;k<kk-1;k++) {
    for(m=1;m<mm-1;m++) {
      GIJ=GI(m,k);
      PIJ=PAI(m,k);

      if ( m==1 || m==mm-2 ) {
        xtxx=txx[GIJ]-txx[GI(m-1,k)];
        xtxz=txz[GI(m+1,k)]-txz[GIJ];
      }
      else {
      	xtxx = coe1*(txx[GIJ]  -txx[GI(m-1,k)]) + coe2*(txx[GI(m+1,k)]-txx[GI(m-2,k)]);
        xtxz = coe1*(txz[GI(m+1,k)]-txz[GIJ])   + coe2*(txz[GI(m+2,k)]-txz[GI(m-1,k)]);
      }
      
      if( k==1 || k==kk-2 )  {
        ztxz=txz[GIJ]-txz[GI(m,k-1)];
        ztzz=tzz[GI(m,k+1)]-tzz[GIJ];
      }
      else {
        ztxz = coe1*(txz[GIJ]  -txz[GI(m,k-1)]) + coe2*(txz[GI(m,k+1)]-txz[GI(m,k-2)]);
        ztzz=  coe1*(tzz[GI(m,k+1)]-tzz[GIJ])   + coe2*(tzz[GI(m,k+2)]-tzz[GI(m,k-1)]);
      }
      // dtdx = dt/dx 
      vx[GIJ] = vx[GIJ] + (xtxx+ztxz)*2.0*dtdx/(c[m][k][0]+c[m-1][k][0]);
      vz[GIJ] = vz[GIJ] + (xtxz+ztzz)*2.0*dtdx/(c[m][k][0]+c[m][k+1][0]);
      // PML at X direction
      if(m<nabs) {
         ipml=nabs-1-m;
         // vx 
         Ttmp = (2.0*pml_xtxx[PIJ] + pml_d[ipml]*xtxx)/(2.0+pml_alpha[ipml]);
         vx[GIJ] = vx[GIJ] + ((pml_beta[ipml]-1)*xtxx/dx - pml_beta[ipml]*Ttmp)*dt*2.0/(c[m][k][0]+c[m-1][k][0]);
         pml_xtxx[PIJ] = 2.0*Ttmp - pml_xtxx[PIJ];
         // vz
         Ttmp = (2.0*pml_xtxz[PIJ] + pml_d_half[ipml]*xtxz)/(2.0+pml_alpha_half[ipml]);
         vz[GIJ] = vz[GIJ] + ((pml_beta_half[ipml]-1)*xtxz/dx - pml_beta_half[ipml]*Ttmp)*dt*2.0/(c[m][k][0]+c[m][k+1][0]);
         pml_xtxz[PIJ] = 2.0*Ttmp - pml_xtxz[PIJ];
      }
      if(m>=mm-nabs) {
         // vz
         ipml=m-(mm-nabs); 
         Ttmp = (2.0*pml_xtxz[PIJ] + pml_d_half[ipml]*xtxz)/(2.0+pml_alpha_half[ipml]);
         vz[GIJ] = vz[GIJ] + ((pml_beta_half[ipml]-1)*xtxz/dx - pml_beta_half[ipml]*Ttmp)*dt*2.0/(c[m][k][0]+c[m][k+1][0]);
         pml_xtxz[PIJ] = 2.0*Ttmp - pml_xtxz[PIJ];
      }
      if(m>mm-nabs) {
         // vx 
         ipml=m-(mm-nabs)-1;
         Ttmp = (2.0*pml_xtxx[PIJ] + pml_d[ipml]*xtxx)/(2.0+pml_alpha[ipml]);
         vx[GIJ] = vx[GIJ] + ((pml_beta[ipml]-1)*xtxx/dx - pml_beta[ipml]*Ttmp)*dt*2.0/(c[m][k][0]+c[m-1][k][0]);
         pml_xtxx[PIJ] = 2.0*Ttmp - pml_xtxx[PIJ];
      }

      // PML at Z direction
      if(k<nabs_top) {
         ipml=nabs_top-1-k;
         // vx
         Ttmp = (2.0*pml_ztxz[PIJ] + pml_d[ipml]*ztxz)/(2.0+pml_alpha[ipml]);
         vx[GIJ] = vx[GIJ] + ((pml_beta[ipml]-1)*ztxz/dx - pml_beta[ipml]*Ttmp)*dt*2.0/(c[m][k][0]+c[m-1][k][0]);
         pml_ztxz[PIJ] = 2.0*Ttmp - pml_ztxz[PIJ];
         // vz
         Ttmp = (2.0*pml_ztzz[PIJ] + pml_d_half[ipml]*ztzz)/(2.0+pml_alpha_half[ipml]);
         vz[GIJ] = vz[GIJ] + ((pml_beta_half[ipml]-1)*ztzz/dx - pml_beta_half[ipml]*Ttmp)*dt*2.0/(c[m][k][0]+c[m-1][k][0]);
         pml_ztzz[PIJ] = 2.0*Ttmp - pml_ztzz[PIJ];
      }

      if(k>=kk-nabs) {
         // vz
         ipml=k-(kk-nabs); 
         Ttmp = (2.0*pml_ztzz[PIJ] + pml_d_half[ipml]*ztzz)/(2.0+pml_alpha_half[ipml]);
         vz[GIJ] = vz[GIJ] + ((pml_beta_half[ipml]-1)*ztzz/dx - pml_beta_half[ipml]*Ttmp)*dt*2.0/(c[m][k][0]+c[m-1][k][0]);
         pml_ztzz[PIJ] = 2.0*Ttmp - pml_ztzz[PIJ];
      }
      if(k>kk-nabs) {
         // vx
         ipml=k-(kk-nabs)-1;
         Ttmp = (2.0*pml_ztxz[PIJ] + pml_d[ipml]*ztxz)/(2.0+pml_alpha[ipml]);
         vx[GIJ] = vx[GIJ] + ((pml_beta[ipml]-1)*ztxz/dx - pml_beta[ipml]*Ttmp)*dt*2.0/(c[m][k][0]+c[m-1][k][0]);
         pml_ztxz[PIJ] = 2.0*Ttmp - pml_ztxz[PIJ];
      }

  } }
} // end of pragma omp parallel
if (bparams.local_solve){ //If we are performing a local solve, update the velocities around the boundary cells
	  update_boundary_velocity_SSG(boundary_wavefields);
}
 return;
}

/*==================================================================*/
/*  get d(Vx)/dx and d(Vz)/dz at Txx and Tzz, d(Vx)/dz and d(Vz)/dx at Txz */void get_Dv(FILE *fid)
{
#pragma omp parallel default(shared) private(m,k)
{
#pragma omp for
  for(k=1;k<kk-1;k++) {
    for(m=1;m<mm-1;m++) {
       if ( m==1 || m==mm-2 ) {
           Dvxdx[GI(m,k)] = vx[GI(m+1,k)] - vx[GI(m,k)];
           Dvzdx[GI(m,k)] = vz[GI(m,k)] - vz[GI(m-1,k)];
       }
       else {
           Dvxdx[GI(m,k)] = coe1*(vx[GI(m+1,k)]-vx[GI(m,k)]) + coe2*(vx[GI(m+2,k)]-vx[GI(m-1,k)]);  /* for Txx and Tzz, at (m+1/2,k) */
           Dvzdx[GI(m,k)] = coe1*(vz[GI(m,k)]-vz[GI(m-1,k)]) + coe2*(vz[GI(m+1,k)]-vz[GI(m-2,k)]);  /* for Txz, at (m,k+1/2) */
       }

       if( k==1 || k==kk-2 ) {
           Dvzdz[GI(m,k)] = vz[GI(m,k)] - vz[GI(m,k-1)];
           Dvxdz[GI(m,k)] = vx[GI(m,k+1)] - vx[GI(m,k)];
       }
       else {
           Dvzdz[GI(m,k)] = coe1*(vz[GI(m,k)]-vz[GI(m,k-1)]) + coe2*(vz[GI(m,k+1)]-vz[GI(m,k-2)]);   /* for Txx and Tzz, at (m+1/2,k) */
           Dvxdz[GI(m,k)] = coe1*(vx[GI(m,k+1)]-vx[GI(m,k)]) + coe2*(vx[GI(m,k+2)]-vx[GI(m,k-1)]);   /* for Txz, at (m,k+1/2) */
       }
 }}
 for(k=0;k<kk;k++) {
    Dvxdx[GI(0,k)]    = vx[GI(1,k)]    - vx[GI(0,k)];
    Dvzdx[GI(mm-1,k)] = vz[GI(mm-1,k)] - vz[GI(mm-2,k)];
 }
 for(m=0;m<mm;m++) {
    Dvzdz[GI(m,kk-1)] = vz[GI(m,kk-1)] - vz[GI(m,kk-2)];
    Dvxdz[GI(m,0)]    = vx[GI(m,1)]    - vx[GI(m,0)];
 }
} // end of parallel
return;
}





