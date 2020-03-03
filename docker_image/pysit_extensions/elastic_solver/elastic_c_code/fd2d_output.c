/*=================================================================
  * fd2d_output.c -- setup, output snapshots and traces
  *        
  *
  * 2D finite difference modeling seismic waves propagation in fractured media
  * 
  *  Xinding Fang, MIT-ERL, May 2009
  *  Email: xinfang@mit.edu
 =================================================================*/
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "fd2d.h"

/********************************************************************/
FILE *setupss(int whch, int ssind, FILE *fid, double time)
{
  char ssfile[256];
  FILE *SSfid;
  float tmp;

  switch(whch) {
    case 0: sprintf(ssfile,"%s_ssx_%.2f_m_sz_%.2f_m_snapshot%ivx.bin", snaps_output_dir, srcxz[0][0], srcxz[0][1], ssind); break;
    case 1: sprintf(ssfile,"%s_ssx_%.2f_m_sz_%.2f_m_snapshot%ivz.bin", snaps_output_dir, srcxz[0][0], srcxz[0][1], ssind); break;
    case 2: sprintf(ssfile,"%s_ssx_%.2f_m_sz_%.2f_m_snapshot%ipr.bin", snaps_output_dir, srcxz[0][0], srcxz[0][1], ssind); break;
    case 3: sprintf(ssfile,"%s_ssx_%.2f_m_sz_%.2f_m_snapshot%idiv.bin", snaps_output_dir, srcxz[0][0], srcxz[0][1], ssind); break;
    case 4: sprintf(ssfile,"%s_ssx_%.2f_m_sz_%.2f_m_snapshot%icurl.bin", snaps_output_dir, srcxz[0][0], srcxz[0][1], ssind); break;
    case 5: sprintf(ssfile,"%s_ssx_%.2f_m_sz_%.2f_m_snapshot%itxx.bin",  snaps_output_dir, srcxz[0][0], srcxz[0][1], ssind); break;
    case 6: sprintf(ssfile,"%s_ssx_%.2f_m_sz_%.2f_m_snapshot%itzz.bin",  snaps_output_dir, srcxz[0][0], srcxz[0][1], ssind); break;
    case 7: sprintf(ssfile,"%s_ssx_%.2f_m_sz_%.2f_m_snapshot%itxz.bin",  snaps_output_dir, srcxz[0][0], srcxz[0][1], ssind); break;
    case 8: sprintf(ssfile,"%s_ssx_%.2f_m_sz_%.2f_m_snapshot%ivpx.bin",  snaps_output_dir, srcxz[0][0], srcxz[0][1], ssind); break;
    case 9: sprintf(ssfile,"%s_ssx_%.2f_m_sz_%.2f_m_snapshot%ivpz.bin",  snaps_output_dir, srcxz[0][0], srcxz[0][1], ssind); break;
    case 10: sprintf(ssfile,"%s_ssx_%.2f_m_sz_%.2f_m_snapshot%ivsx.bin",  snaps_output_dir, srcxz[0][0], srcxz[0][1], ssind); break;
    case 11: sprintf(ssfile,"%s_ssx_%.2f_m_sz_%.2f_m_snapshot%ivsz.bin",  snaps_output_dir, srcxz[0][0], srcxz[0][1], ssind); break;
  }
  if( !(SSfid=fopen(ssfile,"wb")) )  {
    printf("snapshot file %s not opened, whch = %i\n",ssfile,whch); 
    fprintf(fid,"snapshot file %s not opened, whch = %i\n",ssfile,whch);
    return(0);
  }

  tmp=(float)time;
  fwrite(&tmp, sizeof(float),1,SSfid);

  tmp = 1.0*ss.M;   fwrite(&tmp,sizeof(float),1,SSfid);
  for(m=0;m<ss.M;m++) {
     tmp=(float)gx[ss.m0 + m*ss.dm];
     fwrite(&tmp,sizeof(float),1,SSfid);
  }
    
  tmp = 1.0*ss.K;   fwrite(&tmp,sizeof(float),1,SSfid);
  for(k=0;k<ss.K;k++) {
      tmp=(float)gz[ss.k0 + k*ss.dk];
      fwrite(&tmp,sizeof(float),1,SSfid);
  }

  return(SSfid);
}

/****************************************************************************/
/******** write # of receiver and receiver coordinates **********************/
void hist_setup(double **rcvxz,FILE *sgramfid)
{
  int i;
  float tmp;
  
  tmp=2.0;
  fwrite(&tmp,sizeof(float),1,sgramfid);    // ID for 2D output
  tmp=(float)dx;
  fwrite(&tmp,sizeof(float),1,sgramfid);     /* dx */
  tmp=(float)(hist.di*dt);
  fwrite(&tmp,sizeof(float),1,sgramfid);     /* dt of output trace */
  tmp = (float) itimestep;
  fwrite(&tmp,sizeof(float),1,sgramfid);    /* number of time step */
  tmp = (float) nhist;
  fwrite(&tmp,sizeof(float),1,sgramfid);    /* number of receivers */

  for(i=0;i<nhist;i++) {
    tmp=(float)rcvxz[0][i];
    fwrite(&tmp,sizeof(float),1,sgramfid);
    tmp=(float)rcvxz[1][i];
    fwrite(&tmp,sizeof(float),1,sgramfid);
  }
  tmp = (float) nsrc;
  if(isourcecomp==3) tmp = 0.;  // plane wave
  fwrite(&tmp,sizeof(float),1,sgramfid);   // number of source 
  if(isourcecomp!=3) {
    for(i=0;i<nsrc;i++) {
      tmp=(float)srcxz[0][i];
      fwrite(&tmp,sizeof(float),1,sgramfid);
      tmp=(float)srcxz[1][i];
      fwrite(&tmp,sizeof(float),1,sgramfid);
    }
  }
}


/******************************************/
 double divergence(int m,int k)
{
 /* calculate the divergence at (m+1/2, k) */
 double out;
 out = 0.;
 if (m==0)          out = out + (vx[GI(m+1,k)] - vx[GI(m,k)])/dx;
 else if (m==mm-2)  out = out + (vx[GI(m+1,k)] - vx[GI(m,k)])/dx;
 else if (m==mm-1)  out = out + (vx[GI(m,k)] - vx[GI(m-1,k)])/dx;
 else               out = out + (coe1*(vx[GI(m+1,k)] - vx[GI(m,k)])+coe2*(vx[GI(m+2,k)] - vx[GI(m-1,k)]))/dx;

 if (k==0)          out = out + (vz[GI(m,k+1)] - vz[GI(m,k)])/dx;
 else if (k==1)     out = out + (vz[GI(m,k)] - vz[GI(m,k-1)])/dx;
 else if (k==kk-1)  out = out + (vz[GI(m,k)] - vz[GI(m,k-1)])/dx;
 else               out = out + (coe1*(vz[GI(m,k)] - vz[GI(m,k-1)])+coe2*(vz[GI(m,k+1)] - vz[GI(m,k-2)]))/dx;
 return(out);
}

/******************************************/
 double curl(int m1,int k1)
{
/* calculate the curl of the velocity field at (m, k+1/2) */ 
/*  curl(u)=(dux/dz-duz/dx)ey   */

int ii, jj, m, k;
double tmp[2][2], out, t1, t2;
for(ii=0;ii<=1;ii++) {
   for(jj=-1;jj<=0;jj++) {
      m=m1+ii; k=k1+jj;
      if (k==0)          t1 = (vx[GI(m,k+1)] - vx[GI(m,k)])/dx;
      else if (k==kk-2)  t1 = (vx[GI(m,k+1)] - vx[GI(m,k)])/dx;
      else if (k==kk-1)  t1 = (vx[GI(m,k)] - vx[GI(m,k-1)])/dx;
      else               t1 = (coe1*(vx[GI(m,k+1)] - vx[GI(m,k)])+coe2*(vx[GI(m,k+2)] - vx[GI(m,k-1)]))/dx;

      if (m==0)          t2 = (vz[GI(m+1,k)] - vz[GI(m,k)])/dx;
      if (m==1)          t2 = (vz[GI(m,k)] - vz[GI(m-1,k)])/dx;
      else if (m==mm-1)  t2 = (vz[GI(m,k)] - vz[GI(m-1,k)])/dx;
      else               t2 = (coe1*(vz[GI(m,k)] - vz[GI(m-1,k)])+coe2*(vz[GI(m+1,k)] - vz[GI(m-2,k)]))/dx;
      tmp[ii][jj+1] = (t1-t2);
}}
out=0.25*(tmp[0][0]+tmp[0][1]+tmp[1][0]+tmp[1][1]);

return(out);

}

/********************************************************************
 *  get the appropriate components for hist and snapshot outputs
 ********************************************************************/

double output_var(int ssi, int m, int k)
{
  int gi;
  double d1, d2, d3, d4;
  double out, t1, t2;

  gi = GI(m,k);
  
  switch (ssi) {
    case 0: 
       if(iGRID==0)  out = 0.5*(vx[GI(m,k)]+vx[GI(m+1,k)]);
       else          out = vx[gi];
       break;
    case 1: 
       if(iGRID==0)  out = 0.5*(vz[GI(m,k)]+vz[GI(m,k-1)]);
       else          out = vz[gi]; 
       break;
    case 2: /* calculate the pressure */
       out = 0.5*(txx[gi]+tzz[gi]); //Added by Bram. Need to average. If a pure pressure field, both txx and tzz are equal (making very clear you need the 0.5 to undo the doubling)
       break;
      
    case 3: /* calculate the divergence at (m+1/2, k) */
       out=divergence(m,k);
/*
  out=1.0/dx*(2.0/3.0*( (txx[GI(m,k+1)]+tzz[GI(m,k+1)])-(txx[GI(m,k-1)]+tzz[GI(m,k-1)]) )
     -1.0/12.0*( (txx[GI(m,k+2)]+tzz[GI(m,k+2)])-(txx[GI(m,k-2)]+tzz[GI(m,k-2)]) ));
*/
       break;
      
    case 4: /* calculate the curl of the velocity field at (m, k+1/2) */ 
      /*  y curl */ 
       out=curl(m,k);
       break;
      
    case 5: /* txx */
       out=txx[gi]; break;


    case 6: /* tzz */
       out=tzz[gi]; break;

    case 7: /* txz */
       out=txz[gi]; break;

    case 8: // P wave vpx, div(div(v))_x
// du(x)/dx=[2/3 * (u(x+dx)-u(x-dx)) - 1/12 * (u(x+2dx)-u(x-2dx))]/dx
       if(m>1 && m<mm-2 && k>1 && k<kk-2) {
          d1=divergence(m+1,k);
          d2=divergence(m-1,k);
          d3=divergence(m+2,k);
          d4=divergence(m-2,k);
          out=(2.0/3.0*(d1-d2)-1.0/12.0*(d3-d4))/dx; 
       }
       else 
          out=0.0;
       break;

    case 9: // P wave vpz, div(div(v))_z
       if(m>1 && m<mm-2 && k>1 && k<kk-2) {
          d1=divergence(m,k+1);
          d2=divergence(m,k-1);
          d3=divergence(m,k+2);
          d4=divergence(m,k-2);
          out=(2.0/3.0*(d1-d2)-1.0/12.0*(d3-d4))/dx;
       }
       else 
          out=0.0;
       break;

    case 10: // S wave vsx,  curl(curl(v))_x
       if(m>1 && m<mm-2 && k>1 && k<kk-2) {
          d1=curl(m,k+1);
          d2=curl(m,k-1);
          d3=curl(m,k+2);
          d4=curl(m,k-2);
          out=-(2.0/3.0*(d1-d2)-1.0/12.0*(d3-d4))/dx;
       }
       else
          out=0.0;
       break;

    case 11: // S wave vsz,  curl(curl(v))_z
       if(m>1 && m<mm-2 && k>1 && k<kk-2) {
          d1=curl(m+1,k);
          d2=curl(m-1,k);
          d3=curl(m+2,k);
          d4=curl(m-2,k);
          out=(2.0/3.0*(d1-d2)-1.0/12.0*(d3-d4))/dx;
       }
       else 
          out=0.0;
       break;

    } /* end of switch */ 

 return(out);
}


