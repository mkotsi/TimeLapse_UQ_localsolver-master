/*=================================================================
  * fd2d_source.c -- source - default as Ricker wavelet
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
#include "funlist.h"
/********************************************************************/
double source_xz(int istep, FILE *fid)
{
  int isx, isz;
  double force,dtemp,tempx,tempz,time;
  force = 0.;

  time  = dt*istep;

  switch (iwavelet) {
     case 0:  /* Ricker wavelet */
         force = Ricker(time,amp0,freq0);
         break; 
     case 1: /* sine function (single frequency wavelet) */
         force = sine(time,amp0,freq0); 
         break;
     case 2: /* read from file: swl.asc  */
         i=floor(istep*dt/dtswl);
         j=i+1;
         if(i>=nswl) i=nswl-1;
         if(j>=nswl) j=nswl-1;
         if(i==j) force = swl[nswl-1];
         else     force = swl[j]+(swl[i]-swl[j])*(istep*dt-j*dtswl)/(i-j)/dtswl;
         break;
     case 3: /* Get from array passed from Python */
    	 tsrcf = itimestep;
    	 force = amp0*source_arr[istep];
    	 break;
     case 9: //delta
    	 tsrcf = itimestep;
    	 if (istep==0) {
    		 force = amp0;
    	 }
    	 else {
    		 force = 0;
    	 }
    	 break;
  }
  force = force*dt/(dx*dx);

if(isourcecomp!=3) { // not planewave
  for (j=0;j<nsrc;j++) {
      isx = srcind[0][j];
      isz = srcind[1][j];

//time=istep*dt-0.008715574274766/1000*j*dx;
//if(time>0.0) force = Ricker(time,amp0,freq0); 
//else  force=0.0;
      switch(isourcecomp) {
        case 0: /*  add compressional source at [GI(isx,isy,isz)] */
          if(iGRID==0 || iGRID==3) { // SSG
             txx[GI(isx,isz)] = txx[GI(isx,isz)] + force;
             tzz[GI(isx,isz)] = tzz[GI(isx,isz)] + force;
          }
          else {  // RSG
             force = 0.5*force;
             for(k=-1;k<=0;k++) {
               for(m=-1;m<=0;m++) {
                 txx[GI(isx+m,isz+k)] = txx[GI(isx+m,isz+k)] + force;
                 tzz[GI(isx+m,isz+k)] = tzz[GI(isx+m,isz+k)] + force;
             }}
          }
          break;

        case 1:/*  add seismic moment tensor at [isx][isy][isz].*/
          if(iGRID==0 || iGRID==3) {  //SSG
            tempx = sm[0][0]*force;/*  add Mxx, Mzz */
            tempz = sm[1][1]*force;
            vx[GI(isx,isz)]   = vx[GI(isx,isz)]   - tempx;
            vz[GI(isx,isz)]   = vz[GI(isx,isz)]   + tempz;
            vx[GI(isx+1,isz)] = vx[GI(isx+1,isz)] + tempx;
            vz[GI(isx,isz-1)] = vz[GI(isx,isz-1)] - tempz;
          
            tempx = sm[0][1]*force;      /*  add Mxz */
            vx[GI(isx,isz+1)]   = vx[GI(isx,isz+1)]   + tempx;
            vx[GI(isx,isz-1)]   = vx[GI(isx,isz-1)]   - tempx;
            vx[GI(isx+1,isz+1)] = vx[GI(isx+1,isz+1)] + tempx;
            vx[GI(isx+1,isz-1)] = vx[GI(isx+1,isz-1)] - tempx;
          
            tempz = sm[1][0]*force;      /*  add Mzx */
            vz[GI(isx+1,isz)]   = vz[GI(isx+1,isz)]   + tempz;
            vz[GI(isx-1,isz)]   = vz[GI(isx-1,isz)]   - tempz;
            vz[GI(isx+1,isz-1)] = vz[GI(isx+1,isz-1)] + tempz;
            vz[GI(isx-1,isz-1)] = vz[GI(isx-1,isz-1)] - tempz;
          }
          else { // RSG

          }
          break;

        case 2: /* vector force, in direction fdir */
          if(iGRID==0 || iGRID==3) {  // SSG
            dtemp = force * c[isx][isz][0]/dx/dx;
            tempx = dtemp * sin(fdir);
            tempz = dtemp * cos(fdir);
        
            vx[GI(isx,isz)]   = vx[GI(isx,isz)]   + tempx;
            vz[GI(isx,isz)]   = vz[GI(isx,isz)]   + tempz;
          }
          else {  // RSG

          }
          break;
      } /* switch(isourcecomp) */
 } // for (j=0;j<nsrc;j++) 
} // if(isourcecomp!=3)
else { // plane wave, add a plane wave in the isdir-direction 
   if(isdir==0) {  // x-direction
     for (k=5;k<kk-5;k++)  txx[GI(nabs+5,k)] = txx[GI(nabs+5,k)] + force;
    }
   else {  // z-direction
     for (m=5;m<mm-5;m++)  tzz[GI(m,nabs_top+5)] = tzz[GI(m,nabs_top+5)] + force;
    //for (m=0;m<mm;m++)  tzz[GI(m,710)] = tzz[GI(m,710)] + force;
   }
}
return(force);
}

/********************************************************************/
/*  Ricker */
double Ricker(double time,double amp0,double freq0)
{
  double force,a;
  //time = time - ( 1.5/freq0 ); // delay
  a = freq0*sqrt(2.)*pi;
  time = time - (6*1/a); //Edit Bram: Same as in PySIT. delay is 6*sigma = 6*1/a.
  tsrcf = 1000000000; //In PySIT I am using threshold = 0.0 (so the source is never cut off. I am doing the same here now by putting the cutoff time to unreasonably large values which we will never exceed)
  if (time>tsrcf)   force = 0.;
  else              force=amp0*a*a*time*exp(-a*a*time*time/2); // integral of Ricker function
  //force=-amp0*a*a*a*a*time*(3-a*a*time*time)*exp(-a*a*time*time/2);
  //else              force = amp0*a*a*(a*a*time*time - 1)*exp(-a*a*time*time/2);
  return(force);
}
 
/********************************************************************/
/*  sine wave, single frequency */
/* f = A*sin(2*pi*f0*t) */
double sine(double time,double amp0,double freq0)
{
  double force;
  time  = time - ( 0.5/freq0 );
  force = 0;
  if ( time <= 0.5/freq0 )   force = amp0 * sin(2. * pi * freq0 * time);
  return(force);
}

