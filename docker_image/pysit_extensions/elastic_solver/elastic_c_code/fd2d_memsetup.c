/*=================================================================
  * fd2d_update_i.c -- 
  *        void setup(FILE *fid)   -- setup and initialize basic variables
  *        void cleanup(FILE *fid) -- cleanup all variables
  *        void cleanup_early(FILE *fid,int when) -- cleanup memory when encounter ERROR
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
#include "funlist.h"

/***************************************************************/
void setup(FILE *fid)
{
  txx   = d1matrix(mm*kk);
  tzz   = d1matrix(mm*kk);  
  txz   = d1matrix(mm*kk);  
  vx    = d1matrix(mm*kk);  
  vz    = d1matrix(mm*kk);  

  if(iGRID==0) {
    Dvxdx = d1matrix(mm*kk);
    Dvzdz = d1matrix(mm*kk);
    Dvxdz = d1matrix(mm*kk);
    Dvzdx = d1matrix(mm*kk);
  }

 // PML damping values 
  if(nabs>0) {
     pml_beta = d1matrix(nabs);
     pml_alpha = d1matrix(nabs);
     pml_d = d1matrix(nabs);
  
     pml_beta_half = d1matrix(nabs);
     pml_alpha_half = d1matrix(nabs);
     pml_d_half = d1matrix(nabs);

     pml_vxx = d1matrix(nelm);
     pml_vzz = d1matrix(nelm);
     pml_vxz = d1matrix(nelm);
     pml_vzx = d1matrix(nelm);
     pml_xtxx = d1matrix(nelm);
     pml_ztzz = d1matrix(nelm);
     pml_xtxz = d1matrix(nelm);
     pml_ztxz = d1matrix(nelm);
  }

  fprintf(fid,"memory allocation complete.\n");

  for(i=0;i<nabs;i++) {
     pml_beta[i]       = 0.;
     pml_alpha[i]      = 0.;
     pml_d[i]          = 0.;
     pml_beta_half[i]  = 0.;
     pml_alpha_half[i] = 0.;
     pml_d_half[i]     = 0.;
  }

  for(k=0;k<kk;k++) {
    for(m=0;m<mm;m++) {
      vx[GI(m,k)]  = 0.;
      vz[GI(m,k)]  = 0.;
      txx[GI(m,k)] = 0.;
      tzz[GI(m,k)] = 0.;
      txz[GI(m,k)] = 0.;
      if(iGRID==0) {
        Dvxdx[GI(m,k)] = 0.;
        Dvzdz[GI(m,k)] = 0.;
        Dvxdz[GI(m,k)] = 0.;
        Dvzdx[GI(m,k)] = 0.;
      }

      if(nabs>0) {
         if(m<nabs || m>=mm-nabs || k<nabs_top || k>=kk-nabs ) {    
            pml_vxx[PAI(m,k)]  = 0.;
            pml_vzz[PAI(m,k)]  = 0.;
            pml_vxz[PAI(m,k)]  = 0.;
            pml_vzx[PAI(m,k)]  = 0.;
            pml_xtxx[PAI(m,k)]  = 0.;
            pml_ztzz[PAI(m,k)]  = 0.;
            pml_xtxz[PAI(m,k)]  = 0.;
            pml_ztxz[PAI(m,k)]  = 0.;
    	 }
      }
  }}
 fprintf(fid,"Initialize variables complete. \n");
}
/***************************************************************/
void cleanup(FILE *fid)
{
  free_1d(gx); free_1d(gz);
  if(modeltype==0) {
     free_1d(zlayer);
     free_2d(co);
  }
  free_3d(c); 
  if(iGRID!=3) free_2i(ipro);

  if(NDF>0) {
     if(DFprop==0) {
        free_1d(ZN);  free_1d(ZT); free_2d(cani); 
     }
     else {
        free_1d(VP_DF); free_1d(VS_DF); free_1d(dens_DF);
     }
     free_2d(DFxz);
 }
 if (isourcecomp!=3) {
   if (isourcecomp==1) free_2d(sm);
   free_2i(srcind);
   //free_2d(srcxz); #These are just pointing to arrays allocated in Python now
 }
 if (hist.l) {
  free_2i(rcvind);
  //free_2d(rcvxz); #These are just pointing to arrays allocated in Python now
  free_1f(rcvval);
  }
  
  free_1d(txx); free_1d(tzz);  free_1d(txz); 
  free_1d(vx);  free_1d(vz);

  if(iGRID==0) {
     free_1d(Dvxdx); free_1d(Dvzdz); 
     free_1d(Dvxdz); free_1d(Dvzdx);
  }

  if(nabs>0) {
     free_1d(pml_beta);      free_1d(pml_alpha);      free_1d(pml_d); 
     free_1d(pml_beta_half); free_1d(pml_alpha_half); free_1d(pml_d_half); 
     free_1d(pml_vxx);  free_1d(pml_vzz);  free_1d(pml_vxz);  free_1d(pml_vzx); 
     free_1d(pml_xtxx); free_1d(pml_ztzz); free_1d(pml_xtxz); free_1d(pml_ztxz); 
  }

  fprintf(fid,"memory clean up complete.\n");

}

/****************************************************************/
void cleanup_early(FILE *fid,int when)
{
  if ( when==1 ) return;
  
  free_1d(gx);  free_1d(gz);
  if ( when==2 ) return;

  if(modeltype==0) {
     free_1d(zlayer);
     free_2d(co);
  }

 if (when==3) return;

   if (NDF>0) {
     free_2d(DFxz);
     if(DFprop==0) {
      free_1d(ZN);  free_1d(ZT);
     }
     else {
      free_1d(VP_DF); free_1d(VS_DF); free_1d(dens_DF);
     }
   }
  if ( when==4 ) return;
  
  if(isourcecomp!=3) {
    free_2i(srcind);
    free_2d(srcxz);
  }
  if (when==5) return; 

  if (isourcecomp!=3 && isourcecomp==1)  free_2d(sm);
  if ( when==6 ) return;

 if (hist.l) {
   free_1f(rcvval);
   free_2d(rcvxz);
   free_2i(rcvind);
 }
return;
}  
    

/****************************************************************/
int    *i1matrix(int M)
{
  int *pt1;
  pt1=(int *) malloc((M*sizeof(int)));
  if (!pt1) printf("allocation failure 2 in dmatrix()");
  //memset(pt1, 0, (M*sizeof(int)));
  return pt1;
}

/****************************************************************/
int	**i2matrix(int M, int N)
{
  int i,j;
  int **pt2, *pt1;
	
  pt1=(int *) malloc((M*N*sizeof(int)));
	if (!pt1) printf("allocation failure 2 in i2matrix()");
  //memset(pt1, 0, (M*N*sizeof(int)));

  pt2=(int **) malloc((M*N*sizeof(int*)));
  if (!pt2) printf("allocation failure in i2matrix()");

  pt2[0]=pt1;
  for(i=1,j=1;i<M;i++,j++) 
      pt2[j]=pt2[j-1] + N;
	
  return pt2;
}

/****************************************************************/
int	***i3matrix(int M, int N, int K)
{
  int i,j;
  int ***pt3, **pt2, *pt1;
	
  pt1=(int *) malloc((M*N*K*sizeof(int)));
	if (!pt1) printf("allocation failure 2 in i3matrix()");
  //memset(pt1, 0, (M*N*K*sizeof(int)));

  pt2=(int **) malloc((M*N*sizeof(int*)));
  if (!pt2) printf("allocation failure in i3matrix()");

  pt3=(int ***) malloc((M*sizeof(int*)));
  if (!pt3) printf("allocation failure in i3matrix()");

  pt2[0]=pt1;
  for(i=1,j=1;i<M*N;i++,j++) 
      pt2[j]=pt2[j-1] + K;

  pt3[0]=pt2;
  for(i=1,j=1;i<M;i++,j++) 
      pt3[j]=pt3[j-1] + N;
	
  return pt3;
}

/****************************************************************/
double	*d1matrix(int M)
{
  double *pt1;
  pt1=(double *) malloc((M*sizeof(double)));
  if (!pt1) printf("allocation failure 2 in dmatrix()");
  //memset(pt1, 0,  (M*sizeof(double)));
  return pt1;
}

/****************************************************************/
double	**d2matrix(int M, int N)
{
  int i,j;
  double **pt2, *pt1;
	
  pt1=(double *) malloc((M*N*sizeof(double)));
  if (!pt1) printf("allocation failure 2 in dmatrix()");
  //memset(pt1, 0,  (M*N*sizeof(double)));

  pt2=(double **) malloc( (M*N*sizeof(double*)));
  if (!pt2) printf("allocation failure in dmatrix()");

  pt2[0]=pt1;
  for(i=1,j=1;i<M;i++,j++) 
      pt2[j]=pt2[j-1] + N;
	
  return pt2;
}

/****************************************************************/
double	***d3matrix(int M, int N, int K)
{
  int i,j;
  double ***pt3, **pt2, *pt1;
	
	/* allocate pointers */
  pt1=(double *) malloc((M*N*K*sizeof(double)));
  if (!pt1) printf("allocation failure 2 in dmatrix()");
  //memset(pt1, 0, (size_t) (M*N*K*sizeof(double)));

  pt2=(double **) malloc( (M*N*sizeof(double*)));
  if (!pt2) printf("allocation failure in dmatrix()");

  pt3=(double ***) malloc((M*sizeof(double*)));
  if (!pt3) printf("allocation failure in dmatrix()");

  pt2[0]=pt1;
  for(i=1,j=1;i<M*N;i++,j++) 
      pt2[j]=pt2[j-1] + K;

  pt3[0]=pt2;
  for(i=1,j=1;i<M;i++,j++) 
      pt3[j]=pt3[j-1] + N;
	
  return pt3;
}

/****************************************************************/
float	*f1matrix(int M)
{
  float *pt1;
  pt1=(float *) malloc((M*sizeof(float)));
  if (!pt1) printf("allocation failure 2 in dmatrix()");
  //memset(pt1, 0,  (M*sizeof(float)));
  return pt1;
}


/****************************************************************/
void free_1d(double *ptr)
{	
  free(ptr);
  return;
}

void free_1f(float *ptr)
{	
  free(ptr);
  return;
}

void free_1i(int *ptr)
{	
  free(ptr);
  return;
}
/****************************************************************/
void free_2d(double **ptr)
{	  free(ptr[0]); free(ptr);
  return;
}

void free_2i(int **ptr)
{	  free(ptr[0]); free(ptr);
  return;
}
/****************************************************************/
void free_3d(double ***ptr)
{	
   free(ptr[0][0]);  free(ptr[0]);  free(ptr);
   return;
}
void free_3i(int ***ptr)
{	
   free(ptr[0][0]);  free(ptr[0]);  free(ptr);
   return;
}


