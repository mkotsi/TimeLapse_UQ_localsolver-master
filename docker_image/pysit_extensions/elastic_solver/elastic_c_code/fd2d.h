 
/*=================================================================
  * fd2d.h -- include global variables in this code
  *
  * 2D finite difference modeling seismic waves propagation in fractured media
  * 
  *  Xinding Fang, MIT-ERL, May 2009
  *  Email: xinfang@mit.edu
 =================================================================*/

#define NOUTVAR    12   // number of output variables for ss and hist 

#define coe1    1.125               // 9/8 = 1.125  
#define coe2    -0.041666666666666  // 1/24 = 0.041666666666666
#define erro    0.0000000001
#define pi      3.141592653589793
#define CI(i,j)   (j>=i)? (((13-i)*i)/2+j-i+1) : (((13-j)*j)/2+i-j+1)  /* Cij */

#define GI(m,k) ((m) + mm*(k)) /* model grid with abs bnds */
#define PXI(m,k)	(  ((k-nabs_top)>=0) && ((k-nabs_top)<KK) && ((m-nabs)>=MM) ? 1:0  )
#define PZI(m,k)	((k-nabs_top)<0 ? 0:   ((k-nabs_top)<KK ? (k-nabs_top):KK)    )
#define PAI(m,k)	(GI(m,k) - (MM*PXI(m,k) + MM*PZI(m,k)))

//local solver defines
#define txx_off  0
#define tzz_off  1
#define txz_off  2
#define  vx_off  3
#define  vz_off  4

 /* io variables: edit Bram */
 char input_file[256]; //get from runtime argument
 char* traces_output_dir;
 char* snaps_output_dir;

 int traces_mem;
 int snaps_mem;

 /* basic useful variables **********************************************/
 int i,j,k,n,m,jj;
 //char inputfile[30];
 //char paramfile[25];
 
 /* basic FD calculation variables **************************************/
 int iGRID;                       /* flag: 0 = SSG
                                           1 = 2nd order RSG
                                           2 = 4th order RSG           */
 double dx, dt;                   /* element size, time step            */
 double gx0,gz0,*gx,*gz;          /* grid coordinates                   */
 int itimestep;                   /* number of time steps               */
 double tfin;                     /* final time                         */
 double dtdx;			  /* dt/dx: used for updating velocities               */
 double *txx, *txz, *tzz;         /* stress tensor components           */   
 double *vx, *vz;                 /* velocity vector components         */
 double *Dvxdx, *Dvzdz, *Dvxdz, *Dvzdx;
 
/* model index  *****************************************************/
 int mm,kk;                      /* model size with abs. boundaries              */
 int MM,KK;                      /* model size without abs. boundaries           */
 int nelm;                       /* number of elements in boundary */
  
/* model  *******************/
 int modeltype;
 double ***c, **co, **cani;      /* elastic constants             */
 int Nani;                       // numer of grid cells with anisotropic elastic constants
 int **ipro;                  // flag: -1=isotropy; >=0 =anisotropy;
 double *zlayer;                 /* depth of the bottom of i-th layer */
 int nlayer;                     /* number of layers - equal to nmat */
 int NDF;                        /* number of discrete fractures */
 int DFprop;             /* flag: 0 = use fracture compliance Zij to represent fracture 
                                  1 = use different VP_DF, VS_DF and dens_DF to represent fracture */
 double *ZN, *ZT;               /* normal compliance and tangential compliance */
 double *VP_DF, *VS_DF, *dens_DF;
 double **DFxz;                 /* coordinates [x,z] of two ends of fractures */
 int iHeter;                    /* flay: =1  add heterogeneous layer
                                         =0  not add         */
 int iheter_top, iheter_bot;  /* top and bottom index of the heterogeneous layer */
 int iheter_th;                 /* thickness of the heterogeneous layer */
 double **ep;                      /* velocity variation */
 /* source variables *********************************************************/
 int nsrc;                /* number of sources to fire simultaneously        */
 double **srcxz;          /* source location in meters                       */
 int **srcind;            /* source location in meters                       */
 int isourcecomp;         /* source type flag     */
 int iwavelet;            /* source wavelet type: 0 = Ricker wavelt; 1 = plane wave */
 double amp0,freq0,tsrcf; /* amplitude, center freq, max time of source wvlt */
 double **sm;             /* moment tensor                                */
 double fdir;             /* force direction, for vector force */
 int isdir;
 int nswl;
 double dtswl;
 double *swl;
 double *source_arr;
 
/* PML absorbing boundary condition */
 int ifsbc;                      /* flag of free surface boundary condition
                                   1: free surface, 0: PML */
 int nabs, nabs_top;  /* thickness of abs. boundaries */
 double p_power, d0factor, PPW0, Vmax;
 double *pml_beta,      *pml_alpha,      *pml_d;
 double *pml_beta_half, *pml_alpha_half, *pml_d_half;
 double *pml_vxx,  *pml_vzz,  *pml_vxz,  *pml_vzx; 
 double *pml_xtxx, *pml_ztzz, *pml_xtxz, *pml_ztxz; 

 /* receiver array ****************************************************/
 double **rcvxz;
 float *rcvval;
 int **rcvind;
 
/* output parameters - snapshots, receivers (hist), and screen ********/ 
 FILE *SSfid;
 struct outparams  {
                  int l;
                  int ii;
                  int ie;
                  int di;
                };

 struct outparams hist;    /* use history points (receivers) */
 struct outparams snap;    /* output snapshots */
 struct outparams screen;  /* screen display */
 int nhist;                /* total number of receivers */
 int histvar[NOUTVAR];

 struct ssparams  {
                    int var[NOUTVAR]; /* flag */
                    int m0;
                    int dm;
                    int mm;
                    int k0;
                    int dk;
                    int kk;
                    int M;
                    int K;
                 };
 struct ssparams ss;
 float *ssval;

 struct boundary_params {
	 int rec_boundary; //bool for whether we want to go into recording mode (in that case, write to  boundary_wavefields and boundary_times).
	 int local_solve;  //bool for whether we want to go in local solve mode (in that case, read from boundary_wavefields and boundary_times). mm, kk, MM and KK, gx0 and gz0 now correspond to submesh.
	 double rec_x_l;   //left boundary x
	 double rec_x_r;   //right boundary x
	 double rec_z_t;   //top boundary z
	 double rec_z_b;   //bot boundary z
 };

 //associated global vars
 struct boundary_params bparams;
 int rec_x_ind_l;
 int rec_x_ind_r;
 int rec_z_ind_t;
 int rec_z_ind_b;

 struct info_struct {
	 int MM;
	 int KK;
	 int nabs;
	 int ifsbc;
	 double d0factor;
	 double PPW0;
	 double p_power;
	 double dx;
	 double dt;
	 int itimestep;
	 double gx0;
	 double gz0;
	 int iwavelet;
	 double amp0;
	 double freq0;
	 int isourcecomp;
	 int nsrc;
	 double* srcx;
	 double* srcz;
	 struct outparams screen;
	 struct outparams snap;
	 struct outparams hist;
	 int histvar[NOUTVAR];
	 struct boundary_params bparams;
	 struct ssparams ss;
	 int nhist;
	 double* rcvx;
	 double* rcvz;
	 int traces_mem;
	 int snaps_mem;
	 char* traces_output_dir;
	 char* snaps_output_dir;
 };
