/*=================================================================
  * fd2d_update_fsbc.c -- update Vx, Vz, Txx, Tzz and Txz on free surface boundary
  * only use SSG
  * 
  * 2D finite difference modeling seismic waves propagation in fractured media
  * 
  *  Xinding Fang, MIT-ERL, May 2009
  *  Email: xinfang@mit.edu
  *
  * Reference:
  * Jozef Kristek,StudiaGeo,46(2002),P355.
 =================================================================*/
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "fd2d.h"
#include "funlist.h"

/***********************************************************************
 *  update stresses on free surface boundary
 *  assume the media is isotropic near surface
 ***********************************************************************/

/* update free surface boundary */
void update_T_fsbc(int istep, FILE *fid) 
{
  int ipml;
  double exx,ezz,ezx,exz,shear_xz,exx0,Vtmp;
  unsigned long GIJ, PIJ;

  // update Txz 
  for(k=0;k<2;k++) { // k=0, 1 
    for(m=1;m<mm-1;m++) { 
       // GI(m,k)=GI(m,k);

        shear_xz = averag(c[m][k][3],c[m-1][k][3],c[m][k+1][3],c[m-1][k+1][3]);
      
        if(m==1||m==mm-2)  ezx = vz[GI(m,k)] - vz[GI(m-1,k)];
        else               ezx = coe1*(vz[GI(m,k)] - vz[GI(m-1,k)]) + coe2*(vz[GI(m+1,k)] - vz[GI(m-2,k)]);

        if(k==0)  exz = -11.0/12.0*vx[GI(m,k)] + 17.0/24.0*vx[GI(m,k+1)] + 3.0/8.0*vx[GI(m,k+2)] - 5.0/24.0*vx[GI(m,k+3)] + 1.0/24.0*vx[GI(m,k+4)];
        else      exz = coe1*(vx[GI(m,k+1)] - vx[GI(m,k)]) + coe2*(vx[GI(m,k+2)] - vx[GI(m,k-1)]);
	
        txz[GI(m,k)] = txz[GI(m,k)] + shear_xz * (exz + ezx);

        // PML in X direction
        if(m<nabs) {
           PIJ=PAI(m,k); 
           ipml=nabs-1-m;
           Vtmp = (2.0*pml_vzx[PIJ] + pml_d[ipml]*ezx)/(2.0 + pml_alpha[ipml]);
           txz[GI(m,k)] = txz[GI(m,k)] + shear_xz * ( (pml_beta[ipml]-1)*ezx - pml_beta[ipml]*Vtmp*dx );
           pml_vzx[PIJ] = 2.0*Vtmp - pml_vzx[PIJ];
        }
        if(m>mm-nabs) {
           PIJ=PAI(m,k);
           ipml=m-(mm-nabs)-1; 
           Vtmp = (2.0*pml_vzx[PIJ] + pml_d[ipml]*ezx)/(2.0 + pml_alpha[ipml]);
           txz[GI(m,k)] = txz[GI(m,k)] + shear_xz * ( (pml_beta[ipml]-1)*ezx - pml_beta[ipml]*Vtmp*dx );
           pml_vzx[PIJ] = 2.0*Vtmp - pml_vzx[PIJ];
        }
  }}
      
  // update Txx & Tzz 
  k = 0;                     
  for(m=1;m<mm-1;m++) {
    //  GI(m,k)=GI(m,k);
      if(m==1||m==mm-2)  exx = vx[GI(m+1,k)] - vx[GI(m,k)];
      else               exx = coe1*(vx[GI(m+1,k)]-vx[GI(m,k)]) + coe2*(vx[GI(m+2,k)]-vx[GI(m-1,k)]);

      txx[GI(m,k)]=txx[GI(m,k)] + (c[m][k][1]-c[m][k][2]*c[m][k][2]/c[m][k][1])*exx;
      tzz[GI(m,k)] = 0.0;

      // PML in X direction
      if(m<nabs) {
         PIJ=PAI(m,k);
         ipml=nabs-1-m;
         Vtmp = (2.0*pml_vxx[PIJ] + pml_d_half[ipml]*exx)/(2.0 + pml_alpha_half[ipml]);
         txx[GI(m,k)] = txx[GI(m,k)] + (c[m][k][1]-c[m][k][2]*c[m][k][2]/c[m][k][1]) * ((pml_beta_half[ipml]-1)*exx-pml_beta_half[ipml]*Vtmp*dx);
         pml_vxx[PIJ] = 2.0*Vtmp - pml_vxx[PIJ];
      }
      if(m>=mm-nabs) {
         PIJ=PAI(m,k);
         ipml=m-(mm-nabs); 
         Vtmp = (2.0*pml_vxx[PIJ] + pml_d_half[ipml]*exx)/(2.0 + pml_alpha_half[ipml]);
         txx[GI(m,k)] = txx[GI(m,k)] + (c[m][k][1]-c[m][k][2]*c[m][k][2]/c[m][k][1]) * ((pml_beta_half[ipml]-1)*exx-pml_beta_half[ipml]*Vtmp*dx);
         pml_vxx[PIJ] = 2.0*Vtmp - pml_vxx[PIJ];
      }
  }

  k = 1;
  for(m=1;m<mm-1;m++) {
    //  GI(m,k)=GI(m,k);

      if(m==1||m==mm-2){
	 exx0 = vx[GI(m+1,k-1)] - vx[GI(m,k-1)];
	 exx  = vx[GI(m+1,k)]   - vx[GI(m,k)];
      }
      else {
	 exx0 = coe1*(vx[GI(m+1,k-1)] - vx[GI(m,k-1)]) + coe2*(vx[GI(m+2,k-1)] - vx[GI(m-1,k-1)]);
	 exx  = coe1*(vx[GI(m+1,k)]   - vx[GI(m,k)])   + coe2*(vx[GI(m+2,k)]   - vx[GI(m-1,k)]);
      }

      ezz = 1.0/22.0*c[m][k-1][2]/c[m][k-1][1]*exx0 
            - 577.0/528.0*vz[GI(m,k-1)] + 201.0/176.0*vz[GI(m,k)] 
            - 9.0/176.0*vz[GI(m,k+1)]   + 1.0/528.0*vz[GI(m,k+2)];

      txx[GI(m,k)] = txx[GI(m,k)] + c[m][k][1]*exx + c[m][k][2]*ezz;
      tzz[GI(m,k)] = tzz[GI(m,k)] + c[m][k][2]*exx + c[m][k][1]*ezz;

      // PML in X direction
      if(m<nabs) {
         PIJ=PAI(m,k);
         ipml=nabs-1-m;
         Vtmp = (2.0*pml_vxx[PIJ] + pml_d_half[ipml]*exx)/(2.0 + pml_alpha_half[ipml]);
         txx[GI(m,k)] = txx[GI(m,k)] + c[m][k][1] * ( (pml_beta_half[ipml]-1)*exx - pml_beta_half[ipml]*Vtmp*dx );
         tzz[GI(m,k)] = tzz[GI(m,k)] + c[m][k][2] * ( (pml_beta_half[ipml]-1)*exx - pml_beta_half[ipml]*Vtmp*dx);
         pml_vxx[PIJ] = 2.0*Vtmp - pml_vxx[PIJ];
      }
      if(m>=mm-nabs) {
         PIJ=PAI(m,k);
         ipml=m-(mm-nabs); 
         Vtmp = (2.0*pml_vxx[PIJ] + pml_d_half[ipml]*exx)/(2.0 + pml_alpha_half[ipml]);
         txx[GI(m,k)] = txx[GI(m,k)] + c[m][k][1] * ( (pml_beta_half[ipml]-1)*exx - pml_beta_half[ipml]*Vtmp*dx );
         tzz[GI(m,k)] = tzz[GI(m,k)] + c[m][k][2] * ( (pml_beta_half[ipml]-1)*exx - pml_beta_half[ipml]*Vtmp*dx);
         pml_vxx[PIJ] = 2.0*Vtmp - pml_vxx[PIJ];
      }
    }
 return;
}

/***********************************************************************
 ***********************************************************************/

//  update velocities along the boundaries 
void update_V_fsbc(int istep, FILE *fid) 
{
 int ipml;
 double xtxx, ztzz, xtxz, ztxz, Ttmp;
 unsigned long GIJ, PIJ;

 for(k=0;k<2;k++) { // k=0, 1                   
    for(m=1;m<mm-1;m++) {
    //  GI(m,k)=GI(m,k);

      if(m==1||m==mm-2) {
         xtxz = txz[GI(m+1,k)]-txz[GI(m,k)];
         xtxx = txx[GI(m,k)]  -txx[GI(m-1,k)];
      }
      else {
         xtxz = coe1*(txz[GI(m+1,k)]-txz[GI(m,k)])   + coe2*(txz[GI(m+2,k)]-txz[GI(m-1,k)]);
         xtxx = coe1*(txx[GI(m,k)]  -txx[GI(m-1,k)]) + coe2*(txx[GI(m+1,k)]-txx[GI(m-2,k)]);
      }
      if(k==0) {
         ztzz = 17.0/24.0*tzz[GI(m,k+1)] + 3.0/8.0*tzz[GI(m,k+2)] - 5.0/24.0*tzz[GI(m,k+3)] + 1.0/24.0*tzz[GI(m,k+4)];
	 ztxz = 35.0/8.0*txz[GI(m,k)] - 35.0/24.0*txz[GI(m,k+1)] + 21.0/40.0*txz[GI(m,k+2)] - 5.0/56.0*txz[GI(m,k+3)];
      }
      else  {
         ztzz = coe1*(tzz[GI(m,k+1)]-tzz[GI(m,k)])+coe2*(tzz[GI(m,k+2)]-tzz[GI(m,k-1)]);
         ztxz = -31.0/24.0*txz[GI(m,k-1)] + 29.0/24.0*txz[GI(m,k)] - 3.0/40.0*txz[GI(m,k+1)] + 1.0/168.0*txz[GI(m,k+2)];
      }

      vx[GI(m,k)] = vx[GI(m,k)] + (xtxx + ztxz)*dtdx*2.0/(c[m][k][0]+c[m-1][k][0]);  
      vz[GI(m,k)] = vz[GI(m,k)] + (xtxz + ztzz)*dtdx*2.0/(c[m][k][0]+c[m][k+1][0]);


      // PML in X direction
      if(m<nabs) {
         PIJ=PAI(m,k);
         ipml=nabs-1-m;
         // vx 
         Ttmp = (2.0*pml_xtxx[PIJ] + pml_d[ipml]*xtxx)/(2.0+pml_alpha[ipml]);
         vx[GI(m,k)] = vx[GI(m,k)] + ((pml_beta[ipml]-1)*xtxx/dx - pml_beta[ipml]*Ttmp)*dt*2.0/(c[m][k][0]+c[m-1][k][0]);
         pml_xtxx[PIJ] = 2.0*Ttmp - pml_xtxx[PIJ];
         // vz
         Ttmp = (2.0*pml_xtxz[PAI(m,k)] + pml_d_half[ipml]*xtxz)/(2.0+pml_alpha_half[ipml]);
         vz[GI(m,k)] = vz[GI(m,k)] + ((pml_beta_half[ipml]-1)*xtxz/dx - pml_beta_half[ipml]*Ttmp)*dt*2.0/(c[m][k][0]+c[m][k+1][0]);
         pml_xtxz[PIJ] = 2.0*Ttmp - pml_xtxz[PIJ];
      }
      if(m>=mm-nabs) {
         PIJ=PAI(m,k);
         // vz
         ipml=m-(mm-nabs); 
         Ttmp = (2.0*pml_xtxz[PIJ] + pml_d_half[ipml]*xtxz)/(2.0+pml_alpha_half[ipml]);
         vz[GI(m,k)] = vz[GI(m,k)] + ((pml_beta_half[ipml]-1)*xtxz/dx - pml_beta_half[ipml]*Ttmp)*dt*2.0/(c[m][k][0]+c[m][k+1][0]);
         pml_xtxz[PIJ] = 2.0*Ttmp - pml_xtxz[PIJ];
      }
      if(m>mm-nabs) {
         PIJ=PAI(m,k);
         // vx 
         ipml=m-(mm-nabs)-1;
         Ttmp = (2.0*pml_xtxx[PIJ] + pml_d[ipml]*xtxx)/(2.0+pml_alpha[ipml]);
         vx[GI(m,k)] = vx[GI(m,k)] + ((pml_beta[ipml]-1)*xtxx/dx - pml_beta[ipml]*Ttmp)*dt*2.0/(c[m][k][0]+c[m-1][k][0]);
         pml_xtxx[PIJ] = 2.0*Ttmp - pml_xtxx[PIJ];
      }
    }
 }
return;
}



