/*=================================================================
  * fd2d_model.c -- mainly used for setup the stiffness matrix Cij at every node
  *
  * 2D finite difference modeling seismic waves propagation in fractured media
  * 
  *  Xinding Fang, MIT-ERL, May 2009
  *  Email: xinfang@mit.edu
  *
  * Reference: 
  * Coates, R.T. and M. Schoenberg, 1995, Finite-difference modeling of faults and fractures: Geophysics
 =================================================================*/
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "fd2d.h"
#include "funlist.h"


/* re-scale elastic constants (c[i][j][k] = c[i][j][k] * dt / dx), except for the density c[i][j][9] */
 int setup_c(FILE *fid, const double * vp, const double * rho, const double * vs)
 {
 int ic, npar, ii, iF, p, q;
 int indx1, indx2, indz1, indz2;
 double x1, z1, x2, z2, s1, s2, s3, s4, n1, n3, n0;
 double xcell[4], zcell[4];
 double x[4], z[4], L, tmp;
 double A[3][3], C2[6][6], BTM0[6][6], BTM0T[6][6], BTM1[6][6], BTM1T[6][6], Mtmp[6][6];
 double mu, lam2mu, lam; 
 double **ctmp, FLmax;
 double rb, delT,delN;
 float tmpf;
 FILE *tmpfid;

 // background elastic modulus 
 switch(modeltype) {
    case 0: // layer model
       for(k=0;k<kk;k++) {
           ic=0; 
           if(iGRID==0 || iGRID==3)  tmp=gz[k];         // SSG
           else                      tmp=gz[k]+0.5*dx;  // RSG

           if(tmp > zlayer[0]) {
              for(i=1;i<nlayer;i++) 
                  if( tmp > zlayer[i-1] && tmp <= zlayer[i] )  { ic=i; break; }
              if( tmp > zlayer[nlayer-1] )   ic=nlayer-1; 
           }
           for(m=0;m<mm;m++) {
               for(npar=0;npar<4;npar++)  c[m][k][npar] = co[ic][npar]; 
               if(iGRID!=3) ipro[m][k]=-1; // isotropic
           }
       }
       break;
    case 2: // read from input arrays
    	printf("STARTING\n");
        Vmax=0.;
        for(k=nabs_top;k<kk-nabs;k++) {
            for(m=nabs;m<mm-nabs;m++) { //iterate over x faster than over z, so one layer at a time
                //The vp, vs and rho arrays only contain interior nodes.
         	    s1 = vp[(k-nabs_top)*MM + (m-nabs)];
         	    s2 = vs[(k-nabs_top)*MM + (m-nabs)];
         	    s3 = rho[(k-nabs_top)*MM + (m-nabs)];
         	    //printf("s1, s2, s3: %f, %f, %f. \n",s1,s2,s3);
                lam2mu=s3*s1*s1;
                mu=s3*s2*s2;
                lam=lam2mu-2*mu;
                c[m][k][0]=s3;
                c[m][k][1]=lam2mu;
                c[m][k][2]=lam;
                c[m][k][3]=mu;
                if(s1>Vmax) Vmax=s1;
                if (c[m][k][0] < 0 || c[m][k][1] < 0 || c[m][k][2] < 0 || c[m][k][3] < 0){
                	printf("Error! Maybe density was assigned to be smaller than 0 or Vs > Vp... \n");
                	printf("z ind: %i, x ind: %i \n\n", k-nabs, m-nabs);
                	printf("dens         : %e \n"  ,c[m][k][0]);
                	printf("lambda + 2 mu: %e \n"  ,c[m][k][1]);
                	printf("lambda       : %e \n"  ,c[m][k][2]);
                	printf("mu           : %e \n\n",c[m][k][3]);
                	return 1;
                }
        }}
        for(k=nabs_top;k<kk-nabs;k++) {
            for(m=0;m<nabs;m++) {
                for(npar=0;npar<4;npar++) c[m][k][npar]=c[nabs][k][npar];
            }
        }

        for(k=nabs_top;k<kk-nabs;k++) {
            for(m=mm-nabs;m<mm;m++) {
                for(npar=0;npar<4;npar++) c[m][k][npar]=c[mm-nabs-1][k][npar];
            }
        }

        for(k=0;k<nabs_top;k++) {
            for(m=0;m<mm;m++) {
                for(npar=0;npar<4;npar++) c[m][k][npar]=c[m][nabs_top][npar];
            }
        }

        for(k=kk-nabs;k<kk;k++) {
            for(m=0;m<mm;m++) {
                for(npar=0;npar<4;npar++) c[m][k][npar]=c[m][kk-nabs-1][npar];
            }
        }

	 if(iGRID!=3) {
	   for(k=0;k<kk;k++) {
		  for(m=0;m<mm;m++) {
			  ipro[m][k]=-1; // isotropic
	   }}
	 }
	 break;
 } // end of switch


/*************************************************
if( !(tmpfid=fopen("vmodel.dat.bin","rb")) ) {      
    fprintf(fid,"Error opening input model file 'fd2dmodel.asc' \n");
    return(100);
}
Vmax=0.;
for(k=nabs;k<kk-nabs;k++) {
    for(m=nabs;m<mm-nabs;m++) {
        fread(&tmpf,sizeof(float),1,tmpfid);
        tmpf=1000.*tmpf;
        //printf("k=%i m=%i v=%g \n",k,m,tmpf);
        if(tmpf>Vmax) Vmax=tmpf;
        lam = c[m][k][0]*tmpf*tmpf;
        c[m][k][1] = lam; c[m][k][2] = lam;  
}}
fclose(tmpfid);

*************************************************/

/* add point scatterer 
for(m=0;m<mm-nabs;m++) {
   if(fabs(gx[m]-0.0)<=1e-5) {indx1=m; break; }
}
for(k=0;k<kk-nabs;k++) {
  if(fabs(gz[k]-0.0)<=1e-5) {indz1=k; break;}
}

printf("center m=%i k=%i \n",indx1,indz1);
p=ceil(0.14/dx); // 1/8 wave length
lam=1470.0*1470.0*1000.0;
for(m=indx1-p;m<=indx1+p;m++) {
   for(k=indz1-p;k<=indz1+p;k++) {
       if(sqrt(gx[m]*gx[m]+gz[k]*gz[k])<=0.1) {
            c[m][k][0]=1000.0;
            c[m][k][1]=lam;
            c[m][k][2]=lam;
            c[m][k][3]=0.0;
       }
}}
*/
 //===========================================================//
 // calculate effective elastic constants in fracture area
 Nani=0;
 if(NDF>0) {
  
    if(DFprop==0) {
       // find the max fracture length
       FLmax=0.;
       for(iF=0;iF<NDF;iF++) { 
           tmp=sqrt((DFxz[iF][2]-DFxz[iF][0])*(DFxz[iF][2]-DFxz[iF][0])+(DFxz[iF][3]-DFxz[iF][1])*(DFxz[iF][3]-DFxz[iF][1]));
           if(FLmax < tmp) FLmax=tmp;
       }

       ic = (int)(NDF*2.0*(FLmax/dx)); // estimate the max number of anisotropic grid cells
       ctmp = d2matrix(ic,22);
       for(p=0;p<ic;p++) 
          for(q=0;q<22;q++) ctmp[p][q]=0.;
    }

    for(iF=0;iF<NDF;iF++) { 
        x1=DFxz[iF][0]; z1=DFxz[iF][1];
        x2=DFxz[iF][2]; z2=DFxz[iF][3];

        indx1=ceil((x1-gx0)/dx)+nabs;
        indx2=ceil((x2-gx0)/dx)+nabs;
        indz1=ceil((z1-gz0)/dx)+nabs_top;
        indz2=ceil((z2-gz0)/dx)+nabs_top;

        if(fabs(x1-x2)<dx)  indx2=indx1;
        if(fabs(z1-z2)<dx)  indz2=indz1;

        // make sure: indx2 >= indx1; indz2 >= indz1;
        if(indx1>indx2) { m=indx1; indx1=indx2; indx2=m; }
        if(indz1>indz2) { k=indz1; indz1=indz2; indz2=k; }

        //    fracture is described by the following straight line function:
        //        (z-z1)/(x-x1)-(z2-z1)/(x2-x1)=0
        //        (z2-z1)*x+(x1-x2)*z+x1*(z1-z2)+z1*(x2-x1)=0
        //        n1*x+n3*z+n0=0
        //    and we can use this function to determine whether a grid cell is intersected by fracture

        n1=z2-z1;
        n3=x1-x2;
        n0=x1*(z1-z2)+z1*(x2-x1);
        tmp=sqrt(n1*n1+n3*n3);
        n1=n1/tmp;  n3=n3/tmp;  n0=n0/tmp;
        if(fabs(n1)<=erro) {
           n1=0.;
           if(n3<0) { n3=-n3; n0=-n0; }
        }
        if(fabs(n3)<=erro) {
           n3=0.;
           if(n1<0) { n1=-n1; n0=-n0; }
        }

        // local to global coordinates rotation matrix
        // assume x1 is normal to the fracture plane in local coordinate
        A[0][0]=n1;     A[0][1]=0.;     A[0][2]=-n3;
        A[1][0]=0.;     A[1][1]=1.;     A[1][2]=0.;
        A[2][0]=n3;     A[2][1]=0.;     A[2][2]=n1;

        // get Bond transformation matrix
        BondTrans(BTM0,BTM1,A);
        for(p=0;p<6;p++) {
            for(q=0;q<6;q++) {
                BTM0T[p][q]=BTM0[q][p];  // BTMT is the transpose of BTM
                BTM1T[p][q]=BTM1[q][p];
        }}

        for(k=indz1;k<=indz2;k++) {
            for(m=indx1;m<=indx2;m++) {

                if(iGRID==0) { // SSG
                   xcell[0]=gx[m];    zcell[0]=gz[k]-0.5*dx;
                   xcell[1]=gx[m+1];  zcell[1]=gz[k]-0.5*dx;
                   xcell[2]=gx[m+1];  zcell[2]=gz[k]+0.5*dx;
                   xcell[3]=gx[m];    zcell[3]=gz[k]+0.5*dx;
                }
                else { // RSG
                   xcell[0]=gx[m];    zcell[0]=gz[k];
                   xcell[1]=gx[m+1];  zcell[1]=gz[k];
                   xcell[2]=gx[m+1];  zcell[2]=gz[k+1];
                   xcell[3]=gx[m];    zcell[3]=gz[k+1];
                }

                s1=n1*xcell[0]+n3*zcell[0]+n0;
                s2=n1*xcell[1]+n3*zcell[1]+n0;
                s3=n1*xcell[2]+n3*zcell[2]+n0;
                s4=n1*xcell[3]+n3*zcell[3]+n0;

                L=0.;
                if( !( (s1>0. && s2>0. && s3>0. && s4>0.) || (s1<0. && s2<0. && s3<0. && s4<0.) ) ) {
                    // this grid cell is intercepted by fracture
                    if( (fabs(n1)<=erro) || (fabs(n3)<=erro) ) L=dx;
                    else if(fabs(n1)>erro && fabs(n3)>erro) {
                       p=0;
                       x[0]=0.; x[1]=0.; x[2]=0.; x[3]=0.;
                       z[0]=0.; z[1]=0.; z[2]=0.; z[3]=0.;

                       tmp=(-n1*xcell[0]-n0)/n3;
                       if(tmp>=zcell[0] && tmp<=zcell[2])  {x[p]=xcell[0]; z[p]=tmp; p++;}
                       tmp=(-n1*xcell[1]-n0)/n3;
                       if(tmp>=zcell[0] && tmp<=zcell[2])  {x[p]=xcell[1]; z[p]=tmp; p++;}
                       tmp=(-n3*zcell[0]-n0)/n1;
                       if(tmp>=xcell[0] && tmp<=xcell[2])  {x[p]=tmp; z[p]=zcell[0]; p++;}
                       tmp=(-n3*zcell[2]-n0)/n1;
                       if(tmp>=xcell[0] && tmp<=xcell[2])  {x[p]=tmp; z[p]=zcell[2]; p++;}

                       if(p>1) {
                          if(sqrt((x[0]-x[1])*(x[0]-x[1])+(z[0]-z[1])*(z[0]-z[1]))<=erro) {
                             if(sqrt((x[0]-x[2])*(x[0]-x[2])+(z[0]-z[2])*(z[0]-z[2]))>erro) {
                                x[1]=x[2]; z[1]=z[2];
                             }
                             else if(sqrt((x[0]-x[3])*(x[0]-x[3])+(z[0]-z[3])*(z[0]-z[3]))>erro) {
                                x[1]=x[3]; z[1]=z[3];
                             }
                             else {
                               fprintf(fid,"ERROR: in setup anisotropic grid cell! \n");
                               free_2d(ctmp);
                               return(7);
                             }
                          }
                          x[0]=x[0]-x[1];
                          z[0]=z[0]-z[1];
                          L=sqrt(x[0]*x[0]+z[0]*z[0]);
                       }
                   }
                }

                // assume there is no crossing between fractures
                if( L>erro ) {  // this grid is crossed by fracture
                   if(DFprop==0) { // use fracture compliance to calculate the effective elastic constants
                      ipro[m][k] = Nani; // anisotropic, map index [m][k] to the index of cani
                      L=dx*dx/L;

                      rb=c[m][k][2]/c[m][k][1];
                      delT=1e-9*ZT[iF]*c[m][k][3]/(L+1e-9*ZT[iF]*c[m][k][3]);
                      delN=1e-9*ZN[iF]*c[m][k][1]/(L+1e-9*ZN[iF]*c[m][k][1]);

                      for(p=0;p<6;p++) for(q=0;q<6;q++) C2[p][q]=0.;

                      C2[0][0]=c[m][k][1]*(1-delN);  C2[0][1]=c[m][k][2]*(1-delN);        C2[0][2]=C2[0][1]; 
                      C2[1][0]=C2[0][1];             C2[1][1]=c[m][k][1]*(1-rb*rb*delN);  C2[1][2]=c[m][k][2]*(1-rb*delN); 
                      C2[2][0]=C2[0][2];             C2[2][1]=C2[1][2];                   C2[2][2]=c[m][k][1]*(1-rb*rb*delN); 
                      C2[3][3]=c[m][k][3];           C2[4][4]=c[m][k][3]*(1-delT);        C2[5][5]=c[m][k][3]*(1-delT); 

                      if(fabs(n3)>erro) { // fracture normal not in x1 direction
                         // matrix transformation: local coordinate (x1 normal to fracture plane) --> global coordinate
                         Matrix_Multiplication(Mtmp,BTM1T,6,6,C2,6,6);  // Mtmp = BTM * C2 
                         //Matrix_Transpose(BTM,6);                     // matrix transpose
                         Matrix_Multiplication(C2,Mtmp,6,6,BTM1,6,6);  // C2 = BTM * C2 * BTM'
                         //Matrix_Transpose(BTM,6);                     // transpose back to its original form for later use
                      }

                      ctmp[Nani][0]=c[m][k][0]; // density
                      ctmp[Nani][1]=C2[0][0];  ctmp[Nani][2]=C2[0][1];  ctmp[Nani][3]=C2[0][2];  
                      ctmp[Nani][4]=C2[0][3];  ctmp[Nani][5]=C2[0][4];  ctmp[Nani][6]=C2[0][5]; 
                      ctmp[Nani][7]=C2[1][1];  ctmp[Nani][8]=C2[1][2];  ctmp[Nani][9]=C2[1][3];  
                      ctmp[Nani][10]=C2[1][4]; ctmp[Nani][11]=C2[1][5]; 
                      ctmp[Nani][12]=C2[2][2]; ctmp[Nani][13]=C2[2][3]; ctmp[Nani][14]=C2[2][4]; ctmp[Nani][15]=C2[2][5];
                      ctmp[Nani][16]=C2[3][3]; ctmp[Nani][17]=C2[3][4]; ctmp[Nani][18]=C2[3][5]; 
                      ctmp[Nani][19]=C2[4][4]; ctmp[Nani][20]=C2[4][5]; 
                      ctmp[Nani][21]=C2[5][5]; 

                      Nani++;
                   }  
                   else { // use soft material to fill the fracture 
                      mu     = dens_DF[iF]*VS_DF[iF]*VS_DF[iF];
                      lam2mu = dens_DF[iF]*VP_DF[iF]*VP_DF[iF];
                      lam    = lam2mu-2.*mu;
                      c[m][k][0] = dens_DF[iF];
                      c[m][k][1] = lam2mu; c[m][k][2] = lam;  c[m][k][3] = mu;  
                   } // end of if(DFprop==0)
                } // end of if(L>erro)
            } // end of for(m=nabs;m<mm-nabs;m++)
        } // end of for(k=indz1;k<=indz2;k++)
    } // end of for(iF=0;iF<NDF;iF++)
    if(DFprop==0) {
       cani = d2matrix(Nani,22);
       for(p=0;p<Nani;p++) 
           for(q=0;q<22;q++) cani[p][q]=ctmp[p][q];
       free_2d(ctmp);
    }

 } // end of if(NDF>0)


/**********************
         mu     = dens_DF[0]*VS_DF[0]*VS_DF[0];
         lam2mu = dens_DF[0]*VP_DF[0]*VP_DF[0];
         lam    = lam2mu-2*mu;
         tmp=90/180*pi;
for(m=0;m<mm;m++) {
  for(k=0;k<kk;k++) {
//    tmp=((gx[m]*cos(tmp)-gz[k]*sin(tmp))-125)*((gx[m]*cos(tmp)-gz[k]*sin(tmp))-125)/1
//        +((gx[m]*sin(tmp)+gz[k]*cos(tmp))-100)*((gx[m]*sin(tmp)+gz[k]*cos(tmp))-100)/100;
//    if(tmp<=1) {
//      if( ((gx[m]-220)*(gx[m]-220)/9 + (gz[k]-150)*(gz[k]-150)/900)<=1) {
      if(gx[m]>=0.55 && gx[m]<=0.55+0.0017 && gz[k]>=0.2 && gz[k]<=0.275) {
         c[m][k][0] = dens_DF[0];
         c[m][k][1] = lam2mu; c[m][k][7] = lam2mu; c[m][k][12] = lam2mu; 
         c[m][k][2] = lam;    c[m][k][3] = lam;    c[m][k][8] = lam;
         c[m][k][16] = mu;    c[m][k][19] = mu;    c[m][k][21] = mu; 
    }
}}

**********************/

 //===================================================================================//
 if(iHeter==1) {
    fprintf(fid,"modify Cij in the heterogeneous layer \n");
    for(k=iheter_top;k<=iheter_bot;k++)
        for(i=0;i<MM;i++)
            for(npar=1;npar<4;npar++)
                c[nabs+i][k][npar] = c[nabs+i][k][npar]*(1+ep[i][k])*(1+ep[i][k]);
    free_2d(ep);
 }

 //===================================================================================//
 // model properties average
 if(iGRID==1 || iGRID==2) { // RSG
    // average density
    ctmp = d2matrix(mm*kk,1);
    for(k=1;k<kk;k++)
        for(m=1;m<mm;m++)
            ctmp[GI(m,k)][0]=0.25*(c[m][k][0]+c[m-1][k][0]+c[m][k-1][0]+c[m-1][k-1][0]);
    for(k=1;k<kk;k++)
        for(m=1;m<mm;m++)
            c[m][k][0]=ctmp[GI(m,k)][0];
    free_2d(ctmp);
 }

 //===================================================================================//
 // convert c for later use
 for(k=0;k<kk;k++) {
     for(m=0;m<mm;m++) {
         if(iGRID==1 || iGRID==2) // RSG
            c[m][k][0]=dtdx/c[m][k][0];  // dens'=dt/dx*1/dens

         for(npar=1;npar<4;npar++)
             c[m][k][npar]=c[m][k][npar]*dtdx;  // cij'=dt/dx*cij
 }}

 if(Nani>0) {
    for(k=0;k<Nani;k++) {
        if(iGRID==1 || iGRID==2) // RSG
           cani[k][0]=dtdx/cani[k][0];

        for(npar=1;npar<22;npar++)
            cani[k][npar]=cani[k][npar]*dtdx;
    }
 }

 fprintf(fid,"elastic moduli rescaled to Cij = Cij *dt/dx (except for the density) \n\n");
 fprintf(fid,"setup Cij complete.\n");
 return(0);
}

/******************************************************************/
double averag(double av,double bv,double cv,double dv) 
/* do shear moduli average *****************/
{
  
  double c55;
  
  if ( av <= erro || bv <= erro || cv <= erro || dv <= erro ) 
    c55 = 0.;
  else {  /*c55=4.0/(1.0/av+1.0/bv+1.0/c+1.0/dv)*/
    c55 = (4.0*av*bv*cv*dv ) / (bv*cv*dv + av*cv*dv + av*bv*dv + av*bv*cv);
  }
  return(c55);
}

/************* calculate the Bond Transformation Matrix *************/
void BondTrans(double M[6][6], double N[6][6], double b[3][3])
{
/*
   M,N: Bond Transformation Matrix (6x6)
   b: rotation matrix (3x3)
   
   Assume C is the original stiffness tensor, C' is the stiffness tensor after rotation,
   then
         C'=M*C*M'  ( M' != inv(M) )
   we have
         M'=inv(N)
   so
         C=N'*C'*N

   Derivation: 
     Stress T=[txx, tyy, tzz, tyz, txz, txy]'
     Strain S=[exx, eyy, ezz, eyz, exz, exy]'
     After coordinate transformation,
       T'=M*T
       S'=N*S
     From 
       T=C*S
     we can obtain
       M*T=M*C*S=T'
       T'=M*C*inv(N)*S'
       T'=C'*S'
       C'=M*C*inv(N)=M*C*M'

   Reference: B.A. Auld, 1990. Acoustic fields and waves in solids: Robert E. Krieger Publishing Company, page 74.

*/

  // M, C'=M*C*M'
   // row #1
   M[0][0]=b[0][0]*b[0][0];      M[0][1]=b[0][1]*b[0][1];      M[0][2]=b[0][2]*b[0][2];  
   M[0][3]=2.0*b[0][1]*b[0][2];  M[0][4]=2.0*b[0][2]*b[0][0];  M[0][5]=2.0*b[0][0]*b[0][1];   
   // row #2
   M[1][0]=b[1][0]*b[1][0];      M[1][1]=b[1][1]*b[1][1];      M[1][2]=b[1][2]*b[1][2];  
   M[1][3]=2.0*b[1][1]*b[1][2];  M[1][4]=2.0*b[1][2]*b[1][0];  M[1][5]=2.0*b[1][0]*b[1][1];
   // row #3
   M[2][0]=b[2][0]*b[2][0];      M[2][1]=b[2][1]*b[2][1];      M[2][2]=b[2][2]*b[2][2];  
   M[2][3]=2.0*b[2][1]*b[2][2];  M[2][4]=2.0*b[2][2]*b[2][0];  M[2][5]=2.0*b[2][0]*b[2][1];  
   // row #4
   M[3][0]=b[1][0]*b[2][0];                  M[3][1]=b[1][1]*b[2][1];                  M[3][2]=b[1][2]*b[2][2];  
   M[3][3]=b[1][1]*b[2][2]+b[1][2]*b[2][1];  M[3][4]=b[1][0]*b[2][2]+b[1][2]*b[2][0];  M[3][5]=b[1][1]*b[2][0]+b[1][0]*b[2][1];  
   // row #5
   M[4][0]=b[2][0]*b[0][0];                  M[4][1]=b[2][1]*b[0][1];                  M[4][2]=b[2][2]*b[0][2];    
   M[4][3]=b[0][1]*b[2][2]+b[0][2]*b[2][1];  M[4][4]=b[0][0]*b[2][2]+b[0][2]*b[2][0];  M[4][5]=b[0][0]*b[2][1]+b[0][1]*b[2][0];  
   // row #6
   M[5][0]=b[0][0]*b[1][0];                  M[5][1]=b[0][1]*b[1][1];                  M[5][2]=b[0][2]*b[1][2];  
   M[5][3]=b[1][1]*b[0][2]+b[0][1]*b[1][2];  M[5][4]=b[0][0]*b[1][2]+b[0][2]*b[1][0];  M[5][5]=b[1][1]*b[0][0]+b[0][1]*b[1][0]; 

  // N
   // row #1
   N[0][0]=b[0][0]*b[0][0];      N[0][1]=b[0][1]*b[0][1];      N[0][2]=b[0][2]*b[0][2];  
   N[0][3]=b[0][1]*b[0][2];      N[0][4]=b[0][2]*b[0][0];      N[0][5]=b[0][0]*b[0][1];   
   // row #2
   N[1][0]=b[1][0]*b[1][0];      N[1][1]=b[1][1]*b[1][1];      N[1][2]=b[1][2]*b[1][2];  
   N[1][3]=b[1][1]*b[1][2];      N[1][4]=b[1][2]*b[1][0];      N[1][5]=b[1][0]*b[1][1];
   // row #3
   N[2][0]=b[2][0]*b[2][0];      N[2][1]=b[2][1]*b[2][1];      N[2][2]=b[2][2]*b[2][2];  
   N[2][3]=b[2][1]*b[2][2];      N[2][4]=b[2][2]*b[2][0];      N[2][5]=b[2][0]*b[2][1];  
   // row #4
   N[3][0]=2.0*b[1][0]*b[2][0];              N[3][1]=2.0*b[1][1]*b[2][1];              N[3][2]=2.0*b[1][2]*b[2][2];  
   N[3][3]=b[1][1]*b[2][2]+b[1][2]*b[2][1];  N[3][4]=b[1][0]*b[2][2]+b[1][2]*b[2][0];  N[3][5]=b[1][1]*b[2][0]+b[1][0]*b[2][1];  
   // row #5
   N[4][0]=2.0*b[2][0]*b[0][0];              N[4][1]=2.0*b[2][1]*b[0][1];              N[4][2]=2.0*b[2][2]*b[0][2];    
   N[4][3]=b[0][1]*b[2][2]+b[0][2]*b[2][1];  N[4][4]=b[0][0]*b[2][2]+b[0][2]*b[2][0];  N[4][5]=b[0][0]*b[2][1]+b[0][1]*b[2][0];  
   // row #6
   N[5][0]=2.0*b[0][0]*b[1][0];              N[5][1]=2.0*b[0][1]*b[1][1];              N[5][2]=2.0*b[0][2]*b[1][2];  
   N[5][3]=b[1][1]*b[0][2]+b[0][1]*b[1][2];  N[5][4]=b[0][0]*b[1][2]+b[0][2]*b[1][0];  N[5][5]=b[1][1]*b[0][0]+b[0][1]*b[1][0];  

  return;
}

/************* Matrix Transpose *************/
void Matrix_Transpose(double M[6][6], int n)
{
/*
  Matrix Transpose
  M: n by n matrix
*/
int iii, jjj;
double tmp;

for(iii=0;iii<n;iii++) {
   for(jjj=iii+1;jjj<n;jjj++) {
     tmp=M[iii][jjj];
     M[iii][jjj]=M[jjj][iii];
     M[jjj][iii]=tmp;
}}

return;
}

/************* Matrix Transpose *************/
void Matrix_Multiplication(double OUT[6][6], double M[6][6], int m1, int m2, double N[6][6], int n1, int n2)
{
/*
  Matrix Multiplication: M*N
  M: n by n matrix
*/
int iii, jjj, kkk;
double tmp;

if(m2==n1) {
   for(iii=0;iii<m1;iii++) {
      for(jjj=0;jjj<n2;jjj++) {
        tmp=0.0;
        for(kkk=0;kkk<m2;kkk++) {
           tmp=tmp+M[iii][kkk]*N[kkk][jjj];
        }
        OUT[iii][jjj]=tmp;
   }}
}
else {
  printf("ERROR: index incorrect!");
}

return;

}

/********* take the inverse of a matrix *************/
int inverse(double A[6][6],int na)
// take the inverse of matrix A
// if the inverse exists, return 0; otherwise, return 1.
 { 
    int i0, j0, k0, *is, *js;
    double d, p;
    is=(int *)malloc(na*sizeof(int));
    js=(int *)malloc(na*sizeof(int));

    for (k0=0; k0<na; k0++) {
       d=0.0;
       for (i0=k0; i0<na; i0++) {
        for (j0=k0; j0<na; j0++) {
            p=fabs(A[i0][j0]);
            if (p>d) { d=p; is[k0]=i0; js[k0]=j0;}
       }}
        if (d<0.0000000001) // printf("err**not inv\n"); 
          { free(is); free(js); return(1);} 
        if (is[k0]!=k0)
          for (j0=0; j0<na; j0++)
            { p=A[k0][j0]; A[k0][j0]=A[is[k0]][j0]; A[is[k0]][j0]=p; }
        if (js[k0]!=k0)
          for (i0=0; i0<na; i0++)
            { p=A[i0][k0]; A[i0][k0]=A[i0][js[k0]]; A[i0][js[k0]]=p; }
        A[k0][k0]=1.0/A[k0][k0];
        for (j0=0; j0<na; j0++)
          if (j0!=k0)  A[k0][j0]=A[k0][j0]*A[k0][k0];
        for (i0=0; i0<na; i0++)
          if (i0!=k0)
            for (j0=0; j0<na; j0++)
              if (j0!=k0)  A[i0][j0]=A[i0][j0]-A[i0][k0]*A[k0][j0];
        for (i0=0; i0<na; i0++)
          if (i0!=k0)  A[i0][k0]=-A[i0][k0]*A[k0][k0];
      }
    for (k0=na-1; k0>=0; k0--) {
      if (js[k0]!=k0)
         for (j0=0; j0<na; j0++)
            { p=A[k0][j0]; A[k0][j0]=A[js[k0]][j0]; A[js[k0]][j0]=p; }
      if (is[k0]!=k0)
         for (i0=0; i0<na; i0++)
            { p=A[i0][k0]; A[i0][k0]=A[i0][is[k0]]; A[i0][is[k0]]=p; }
    }

free(is); free(js);
return(0);
}


