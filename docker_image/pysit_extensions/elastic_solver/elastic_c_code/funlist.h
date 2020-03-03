 
/*=================================================================
  * funlish.h -- list of functions
  *
  * 2D finite difference modeling seismic waves propagation in fractured media
  * 
  *  Xinding Fang, MIT-ERL, May 2009
  *  Email: xinfang@mit.edu
 =================================================================*/

 /*** fd2d_memsetup.c      memory allocation *********/
  void setup(FILE *fid);
  void cleanup(FILE *fid);
  void cleanup_early(FILE *fid,int when);
  int    *i1matrix(int M);
  int	**i2matrix(int M, int N);
  int	***i3matrix(int M, int N, int K);
  double	*d1matrix(int M);
  double	**d2matrix(int M, int N);
  double	***d3matrix(int M, int N, int K);
  float	 *f1matrix(int M);
  void	free_1i(int *ptr);
  void	free_2i(int **ptr);
  void	free_3i(int ***ptr);
  void	free_1d(double *ptr);
  void	free_2d(double **ptr);
  void	free_3d(double ***ptr);
  void  free_1f(float *ptr);

/*** fd2d_model.c        setup model elastic modulus **********/
  int setup_c(FILE *fid, const double * vp, const double * rho, const double * vs);
  double averag(double a,double b,double c,double d);
  void BondTrans(double M[6][6], double N[6][6], double b[3][3]);
  void Matrix_Transpose(double M[6][6], int n);
  void Matrix_Multiplication(double OUT[6][6], double M[6][6], int m1, int m2, double N[6][6], int n1, int n2); 
  int inverse(double A[6][6],int na);

/*** fd2d_readin.c    readin model parameters from file 'fd2dinput.asc' ********/
  //int readin_fd2d(FILE *fid);
  int readin_from_info_struct(struct info_struct*);
  
/*** fd2d_update_SSG.c    update Txx, Tzz, Txz, Vx, Vz in the interior ***************/
  void update_T_SSG(int istep, FILE *fid, double* boundary_wavefields);
  void update_V_SSG(int istep, FILE *fid, double* boundary_wavefields);
  void get_Dv(FILE *fid);

/*** fd2d_update_RSG2.c    ****************/
  void update_T_RSG2(int istep, FILE *fid);
  void update_V_RSG2(int istep, FILE *fid);

/*** fd2d_update_RSG4.c    ****************/
  void update_T_RSG4(int istep, FILE *fid);
  void update_V_RSG4(int istep, FILE *fid);

/*** fd2d_update_SSG_Acoustic.c    ********/
  void update_T_SSG_Acoustic(int istep, FILE *fid);
  void update_V_SSG_Acoustic(int istep, FILE *fid);

/*** fd2d_update_fsbc.c     update Txx, Tzz, Txz, Vx, Vz on free surface boundary condition *********/
  void update_T_fsbc(int istep, FILE *fid);
  void update_V_fsbc(int istep, FILE *fid);

/*** fd2d_pml.c    setup damping values in the PML region *************/
  void setup_damping(FILE *fid);

/*** fd2d_source.c **************************************************/
  double source_xz(int istep,FILE *fid);
  double Ricker(double time,double amp0,double freq0);
  double sine(double time,double amp0,double freq0);

/*** fd2d_output.c    snapshot output ***********/
  FILE *setupss(int whch, int ssind, FILE *fid, double time);
  void hist_setup(double **rcvxyz,FILE *sgramfid);
  double divergence(int m, int k);
  double curl(int m1, int k1);
  double output_var(int ssi, int m, int k);
/*** fd2d_rec_wavefields.c     recording boundary wavefields required for local solver setup later***********/
  void rec_wavefields(double* boundary_wavefields);

/*** fd2d_update_boundaries.c *******/
  void update_boundary_stress_SSG(double* boundary_wavefields);
  void update_boundary_velocity_SSG(double* boundary_wavefields);

/*** get_physical_cpu_count.c ***/
  int get_physical_cpu_count();
