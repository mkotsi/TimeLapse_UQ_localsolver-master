//Bram, feb 2016
//unit cell (m,k) is formed as follows:
//
//  vx         txx/tzz
//  txz        vz
//
// The txx/tzz node in the cell corresponds with gx[m] and gz[k] correspondingly. Other entries are offset by half dx in respective directions

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "fd2d.h"
#include "funlist.h"


//Using same offset variables as in fd2d_rec_wavefields. Somewhat redundant and will result in more lines of code, but easier to check if I am consistent.
void update_boundary_stress_SSG(double* boundary_wavefields){
	unsigned long GIJ;
	int row_increment = 5*(rec_x_ind_r - rec_x_ind_l +1); //For every row we record on the top and bottom we store 5 quantities. tau_xx, tau_zz, tau_xz, vx and vz
	int col_increment = 5*(rec_z_ind_b - rec_z_ind_t +1);
	int time_offset = (jj-1)*(2*3*row_increment + 2*3*col_increment); //2 times 3 cols (left and right) 2 times 3 rows (top and bot). USING (j-1) BECAUSE THE VELOCITY FROM THE PREVIOUS TIMESTEP LOOP ITERATION IS ALWAYS USED TO UPDATE STRESS AT THE NEW TIMESTEP.
	int outer_offset, inner_offset;
	double c19;
	if (jj==0){
		printf("Use OMP when correcting wavefields? Since only boundary perhaps too little work and only overhead?\n");
		printf("Sometimes I evaluate exactly the same expression twice to update txx and tzz. Should probably use intermediate value to avoid repetition.\n");
	}

	/*********************** TOP ***********************/
	/* ONLY DEAL WITH VERTICAL DERIVATIVE CORRECIONS   */
	int side_offset = time_offset; //counts total accumulated offset from previous sides and previous timesteps
	//One layer above boundary (scattered fields)
		k = rec_z_ind_t-1;
		outer_offset = side_offset + (k-(rec_z_ind_t-1))*row_increment;

		//general update
		for (m=rec_x_ind_l+1;m<=rec_x_ind_r-1;m++){
			c19 = averag(c[m][k][3],c[m-1][k][3],c[m][k+1][3],c[m-1][k+1][3]);
			inner_offset = outer_offset + 5*(m-rec_x_ind_l);
			GIJ = GI(m,k);
			txx[GIJ] += -c[m][k][2]*coe2*boundary_wavefields[inner_offset + 1*row_increment + vz_off];
			tzz[GIJ] += -c[m][k][1]*coe2*boundary_wavefields[inner_offset + 1*row_increment + vz_off];
			txz[GIJ] += -c19*coe2*boundary_wavefields[inner_offset + 2*row_increment + vx_off];
		}

		//On leftmost cell only txx and tzz need to be adjusted
		m = rec_x_ind_l;
		inner_offset = outer_offset + 5*(m-rec_x_ind_l);
		GIJ = GI(m,k);
		txx[GIJ] += -c[m][k][2]*coe2*boundary_wavefields[inner_offset + 1*row_increment + vz_off];
		tzz[GIJ] += -c[m][k][1]*coe2*boundary_wavefields[inner_offset + 1*row_increment + vz_off];

		//On rightmost cell only txz needs to be adjusted
		m = rec_x_ind_r;
		inner_offset = outer_offset + 5*(m-rec_x_ind_l);
		GIJ = GI(m,k);
		c19 = averag(c[m][k][3],c[m-1][k][3],c[m][k+1][3],c[m-1][k+1][3]);
		txz[GIJ] += -c19*coe2*boundary_wavefields[inner_offset + 2*row_increment + vx_off];

	//layer on boundary (mixed). Horizontal derivative contributions will be updated by left and right boundary pass
		k = rec_z_ind_t;
		outer_offset = side_offset + (k-(rec_z_ind_t-1))*row_increment;

		//general update
		for (m=rec_x_ind_l+1;m<=rec_x_ind_r-1;m++){
			c19 = averag(c[m][k][3],c[m-1][k][3],c[m][k+1][3],c[m-1][k+1][3]);
			inner_offset = outer_offset + 5*(m-rec_x_ind_l);
			GIJ = GI(m,k);

			txx[GIJ] += -c[m][k][2]*( coe2*boundary_wavefields[inner_offset + 1*row_increment + vz_off]
									 +coe1*boundary_wavefields[inner_offset + 0*row_increment + vz_off]);
			tzz[GIJ] += -c[m][k][1]*( coe2*boundary_wavefields[inner_offset + 1*row_increment + vz_off]
									 +coe1*boundary_wavefields[inner_offset + 0*row_increment + vz_off]);
			txz[GIJ] +=         c19*(-coe1*boundary_wavefields[inner_offset - 0*row_increment + vx_off]
								     -coe2*boundary_wavefields[inner_offset - 1*row_increment + vx_off]);

			//printf("vz +0 SHOULD BE SAME: %e, %e \n",boundary_wavefields[inner_offset + 0*row_increment + vz_off],vz[GIJ]);
		}

		//On leftmost cell only txx and tzz need to be adjusted
		m = rec_x_ind_l;
		inner_offset = outer_offset + 5*(m-rec_x_ind_l);
		GIJ = GI(m,k);
		txx[GIJ] += -c[m][k][2]*( coe2*boundary_wavefields[inner_offset + 1*row_increment + vz_off]
								 +coe1*boundary_wavefields[inner_offset + 0*row_increment + vz_off]);
		tzz[GIJ] += -c[m][k][1]*( coe2*boundary_wavefields[inner_offset + 1*row_increment + vz_off]
								 +coe1*boundary_wavefields[inner_offset + 0*row_increment + vz_off]);

		//On rightmost cell only txz needs to be adjusted
		m = rec_x_ind_r;
		inner_offset = outer_offset + 5*(m-rec_x_ind_l);
		GIJ = GI(m,k);
		c19 = averag(c[m][k][3],c[m-1][k][3],c[m][k+1][3],c[m-1][k+1][3]);
		txz[GIJ] +=         c19*(-coe1*boundary_wavefields[inner_offset + 0*row_increment + vx_off]
							     -coe2*boundary_wavefields[inner_offset - 1*row_increment + vx_off]);

	//One layer below boundary (perturbed fields). Horizontal derivative contributions will be updated by left and right boundary pass
		k = rec_z_ind_t+1;
		outer_offset = side_offset + (k-(rec_z_ind_t-1))*row_increment;

		//general update
		for (m=rec_x_ind_l+1;m<=rec_x_ind_r-1;m++){
			c19 = averag(c[m][k][3],c[m-1][k][3],c[m][k+1][3],c[m-1][k+1][3]);
			inner_offset = outer_offset + 5*(m-rec_x_ind_l);
			GIJ = GI(m,k);

			txx[GIJ] +=  c[m][k][2]*-coe2*boundary_wavefields[inner_offset - 2*row_increment + vz_off];
			tzz[GIJ] +=  c[m][k][1]*-coe2*boundary_wavefields[inner_offset - 2*row_increment + vz_off];
			txz[GIJ] +=         c19*-coe2*boundary_wavefields[inner_offset - 1*row_increment + vx_off];

			//printf("vx +1 SHOULD BE SAME: %e, %e \n",boundary_wavefields[inner_offset + 0*row_increment + vx_off],vx[GIJ]);
			//printf("vz +1 SHOULD BE SAME: %e, %e \n",boundary_wavefields[inner_offset + 0*row_increment + vz_off],vz[GIJ]);
		}

		//On leftmost cell only txx and tzz need to be adjusted
		m = rec_x_ind_l;
		inner_offset = outer_offset + 5*(m-rec_x_ind_l);
		GIJ = GI(m,k);
		txx[GIJ] +=  c[m][k][2]*-coe2*boundary_wavefields[inner_offset - 2*row_increment + vz_off];
		tzz[GIJ] +=  c[m][k][1]*-coe2*boundary_wavefields[inner_offset - 2*row_increment + vz_off];

		//On rightmost cell only txz needs to be adjusted
		m = rec_x_ind_r;
		inner_offset = outer_offset + 5*(m-rec_x_ind_l);
		GIJ = GI(m,k);
		c19 = averag(c[m][k][3],c[m-1][k][3],c[m][k+1][3],c[m-1][k+1][3]);
		txz[GIJ] +=         c19*-coe2*boundary_wavefields[inner_offset - 1*row_increment + vx_off];

	/*********************** RIGHT ***********************/
	/* ONLY DEAL WITH HORIZONTAL DERIVATIVE CORRECIONS   */
	side_offset += 3*row_increment; //counts total accumulated offset from previous sides and previous timesteps
	//One layer right of boundary (scattered fields)
		m = rec_x_ind_r+1;
		outer_offset = side_offset + ((rec_x_ind_r+1)-m)*col_increment;

		//general update
		for (k=rec_z_ind_t+1;k<=rec_z_ind_b-1;k++){
			c19 = averag(c[m][k][3],c[m-1][k][3],c[m][k+1][3],c[m-1][k+1][3]);
			inner_offset = outer_offset + 5*(k-rec_z_ind_t);
			GIJ = GI(m,k);

			txx[GIJ] += c[m][k][1]*coe2*boundary_wavefields[inner_offset + 1*col_increment + vx_off];
			tzz[GIJ] += c[m][k][2]*coe2*boundary_wavefields[inner_offset + 1*col_increment + vx_off];
			txz[GIJ] += c19*coe2*boundary_wavefields[inner_offset + 2*col_increment + vz_off];
		}

		//On topmost cell only txz needs to be adjusted
		k = rec_z_ind_t;
		inner_offset = outer_offset + 5*(k-rec_z_ind_t);
		GIJ = GI(m,k);
		c19 = averag(c[m][k][3],c[m-1][k][3],c[m][k+1][3],c[m-1][k+1][3]);
		txz[GIJ] += c19*coe2*boundary_wavefields[inner_offset + 2*col_increment + vz_off];

		//On botmost cell only txx and tzz need to be adjusted
		k = rec_z_ind_b;
		inner_offset = outer_offset + 5*(k-rec_z_ind_t);
		GIJ = GI(m,k);
		txx[GIJ] += c[m][k][1]*coe2*boundary_wavefields[inner_offset + 1*col_increment + vx_off];
		tzz[GIJ] += c[m][k][2]*coe2*boundary_wavefields[inner_offset + 1*col_increment + vx_off];


	//layer on boundary (mixed)
		m = rec_x_ind_r;
		outer_offset = side_offset + ((rec_x_ind_r+1)-m)*col_increment;

		//general update
		for (k=rec_z_ind_t+1;k<=rec_z_ind_b-1;k++){
			c19 = averag(c[m][k][3],c[m-1][k][3],c[m][k+1][3],c[m-1][k+1][3]);
			inner_offset = outer_offset + 5*(k-rec_z_ind_t);
			GIJ = GI(m,k);

			txx[GIJ] +=  c[m][k][1]*( coe1*boundary_wavefields[inner_offset + 0*col_increment + vx_off]
									 +coe2*boundary_wavefields[inner_offset + 1*col_increment + vx_off]);
			tzz[GIJ] +=  c[m][k][2]*( coe1*boundary_wavefields[inner_offset + 0*col_increment + vx_off]
									 +coe2*boundary_wavefields[inner_offset + 1*col_increment + vx_off]);
			txz[GIJ] +=         c19*( coe1*boundary_wavefields[inner_offset - 0*col_increment + vz_off]
							         +coe2*boundary_wavefields[inner_offset - 1*col_increment + vz_off]);
		}

		//On topmost cell only txz needs to be adjusted
		k = rec_z_ind_t;
		inner_offset = outer_offset + 5*(k-rec_z_ind_t);
		GIJ = GI(m,k);
		c19 = averag(c[m][k][3],c[m-1][k][3],c[m][k+1][3],c[m-1][k+1][3]);
		txz[GIJ] +=         c19*( coe1*boundary_wavefields[inner_offset - 0*col_increment + vz_off]
						         +coe2*boundary_wavefields[inner_offset - 1*col_increment + vz_off]);

		//On botmost cell only txx and tzz need to be adjusted
		k = rec_z_ind_b;
		inner_offset = outer_offset + 5*(k-rec_z_ind_t);
		GIJ = GI(m,k);
		txx[GIJ] +=  c[m][k][1]*( coe1*boundary_wavefields[inner_offset + 0*col_increment + vx_off]
								 +coe2*boundary_wavefields[inner_offset + 1*col_increment + vx_off]);
		tzz[GIJ] +=  c[m][k][2]*( coe1*boundary_wavefields[inner_offset + 0*col_increment + vx_off]
								 +coe2*boundary_wavefields[inner_offset + 1*col_increment + vx_off]);

	//One layer left of boundary  (perturbed fields)
		m = rec_x_ind_r-1;
		outer_offset = side_offset + ((rec_x_ind_r+1)-m)*col_increment;

		//general update
		for (k=rec_z_ind_t+1;k<=rec_z_ind_b-1;k++){
			c19 = averag(c[m][k][3],c[m-1][k][3],c[m][k+1][3],c[m-1][k+1][3]);
			inner_offset = outer_offset + 5*(k-rec_z_ind_t);
			GIJ = GI(m,k);

			txx[GIJ] +=  c[m][k][1]*( coe2*boundary_wavefields[inner_offset - 2*col_increment + vx_off]);
			tzz[GIJ] +=  c[m][k][2]*( coe2*boundary_wavefields[inner_offset - 2*col_increment + vx_off]);
			txz[GIJ] +=         c19*( coe2*boundary_wavefields[inner_offset - 1*col_increment + vz_off]);
		}

		//On topmost cell only txz needs to be adjusted
		k = rec_z_ind_t;
		inner_offset = outer_offset + 5*(k-rec_z_ind_t);
		GIJ = GI(m,k);
		c19 = averag(c[m][k][3],c[m-1][k][3],c[m][k+1][3],c[m-1][k+1][3]);
		txz[GIJ] +=         c19*( coe2*boundary_wavefields[inner_offset - 1*col_increment + vz_off]);

		//On botmost cell only txx and tzz need to be adjusted
		k = rec_z_ind_b;
		inner_offset = outer_offset + 5*(k-rec_z_ind_t);
		GIJ = GI(m,k);
		txx[GIJ] +=  c[m][k][1]*( coe2*boundary_wavefields[inner_offset - 2*col_increment + vx_off]);
		tzz[GIJ] +=  c[m][k][2]*( coe2*boundary_wavefields[inner_offset - 2*col_increment + vx_off]);


	/*********************** BOT ***********************/
	/* ONLY DEAL WITH VERTICAL DERIVATIVE CORRECIONS   */
	side_offset += 3*col_increment; //counts total accumulated offset from previous sides and previous timesteps
	//One layer below boundary (scattered_fields)
		k = rec_z_ind_b+1;
		outer_offset = side_offset + ((rec_z_ind_b+1)-k)*row_increment;

		//general update
		for (m=rec_x_ind_r-1; m>=rec_x_ind_l+1; m--){
			c19 = averag(c[m][k][3],c[m-1][k][3],c[m][k+1][3],c[m-1][k+1][3]);
			inner_offset = outer_offset + 5*(rec_x_ind_r - m);
			GIJ = GI(m,k);

			txx[GIJ] +=  c[m][k][2]*coe2*boundary_wavefields[inner_offset + 2*row_increment + vz_off];
			tzz[GIJ] +=  c[m][k][1]*coe2*boundary_wavefields[inner_offset + 2*row_increment + vz_off];
			txz[GIJ] +=         c19*coe2*boundary_wavefields[inner_offset + 1*row_increment + vx_off];
		}

		//On rightmost cell only txz needs to be adjusted
		m = rec_x_ind_r;
		inner_offset = outer_offset + 5*(rec_x_ind_r - m);
		GIJ = GI(m,k);
		c19 = averag(c[m][k][3],c[m-1][k][3],c[m][k+1][3],c[m-1][k+1][3]);
		txz[GIJ] +=         c19*coe2*boundary_wavefields[inner_offset + 1*row_increment + vx_off];

		//On leftmost cell only txx and tzz needs to be adjusted
		m = rec_x_ind_l;
		inner_offset = outer_offset + 5*(rec_x_ind_r - m);
		GIJ = GI(m,k);
		txx[GIJ] +=  c[m][k][2]*coe2*boundary_wavefields[inner_offset + 2*row_increment + vz_off];
		tzz[GIJ] +=  c[m][k][1]*coe2*boundary_wavefields[inner_offset + 2*row_increment + vz_off];

	//layer on boundary (mixed)
		k = rec_z_ind_b;
		outer_offset = side_offset + ((rec_z_ind_b+1)-k)*row_increment;

		//general update
		for (m=rec_x_ind_r-1; m>=rec_x_ind_l+1; m--){
			c19 = averag(c[m][k][3],c[m-1][k][3],c[m][k+1][3],c[m-1][k+1][3]);
			inner_offset = outer_offset + 5*(rec_x_ind_r - m);
			GIJ = GI(m,k);

			txx[GIJ] +=  c[m][k][2]*( coe1*boundary_wavefields[inner_offset - 0*row_increment + vz_off]
									 +coe2*boundary_wavefields[inner_offset - 1*row_increment + vz_off]);
			tzz[GIJ] +=  c[m][k][1]*( coe1*boundary_wavefields[inner_offset - 0*row_increment + vz_off]
									 +coe2*boundary_wavefields[inner_offset - 1*row_increment + vz_off]);
			txz[GIJ] +=         c19*( coe1*boundary_wavefields[inner_offset + 0*row_increment + vx_off]
						             +coe2*boundary_wavefields[inner_offset + 1*row_increment + vx_off]);
		}

		//On rightmost cell only txz needs to be adjusted
		m = rec_x_ind_r;
		inner_offset = outer_offset + 5*(rec_x_ind_r - m);
		GIJ = GI(m,k);
		c19 = averag(c[m][k][3],c[m-1][k][3],c[m][k+1][3],c[m-1][k+1][3]);
		txz[GIJ] +=         c19*( coe1*boundary_wavefields[inner_offset + 0*row_increment + vx_off]
					             +coe2*boundary_wavefields[inner_offset + 1*row_increment + vx_off]);

		//On leftmost cell only txx and tzz needs to be adjusted
		m = rec_x_ind_l;
		inner_offset = outer_offset + 5*(rec_x_ind_r - m);
		GIJ = GI(m,k);
		txx[GIJ] +=  c[m][k][2]*( coe1*boundary_wavefields[inner_offset - 0*row_increment + vz_off]
								 +coe2*boundary_wavefields[inner_offset - 1*row_increment + vz_off]);
		tzz[GIJ] +=  c[m][k][1]*( coe1*boundary_wavefields[inner_offset - 0*row_increment + vz_off]
								 +coe2*boundary_wavefields[inner_offset - 1*row_increment + vz_off]);

	//One layer above boundary (perturbed fields)
		k = rec_z_ind_b-1;
		outer_offset = side_offset + ((rec_z_ind_b+1)-k)*row_increment;

		//general update
		for (m=rec_x_ind_r-1; m>=rec_x_ind_l+1; m--){
			c19 = averag(c[m][k][3],c[m-1][k][3],c[m][k+1][3],c[m-1][k+1][3]);
			inner_offset = outer_offset + 5*(rec_x_ind_r - m);
			GIJ = GI(m,k);

			txx[GIJ] +=  c[m][k][2]*coe2*boundary_wavefields[inner_offset - 1*row_increment + vz_off];
			tzz[GIJ] +=  c[m][k][1]*coe2*boundary_wavefields[inner_offset - 1*row_increment + vz_off];
			txz[GIJ] +=         c19*coe2*boundary_wavefields[inner_offset - 2*row_increment + vx_off];

		}

		//On rightmost cell only txz needs to be adjusted
		m = rec_x_ind_r;
		inner_offset = outer_offset + 5*(rec_x_ind_r - m);
		GIJ = GI(m,k);
		c19 = averag(c[m][k][3],c[m-1][k][3],c[m][k+1][3],c[m-1][k+1][3]);
		txz[GIJ] +=         c19*coe2*boundary_wavefields[inner_offset - 2*row_increment + vx_off];

		//On leftmost cell only txx and tzz needs to be adjusted
		m = rec_x_ind_l;
		inner_offset = outer_offset + 5*(rec_x_ind_r - m);
		GIJ = GI(m,k);
		txx[GIJ] +=  c[m][k][2]*coe2*boundary_wavefields[inner_offset - 1*row_increment + vz_off];
		tzz[GIJ] +=  c[m][k][1]*coe2*boundary_wavefields[inner_offset - 1*row_increment + vz_off];

	/*********************** LEFT ************************/
	/* ONLY DEAL WITH HORIZONTAL DERIVATIVE CORRECIONS   */
	side_offset += 3*row_increment; //counts total accumulated offset from previous sides and previous timesteps
	//One layer left of boundary  (scattered fields)
		m = rec_x_ind_l-1;
		outer_offset = side_offset + (m - (rec_x_ind_l-1))*col_increment;

		//general update
		for (k = rec_z_ind_b-1; k >= rec_z_ind_t+1; k--){
			c19 = averag(c[m][k][3],c[m-1][k][3],c[m][k+1][3],c[m-1][k+1][3]);
			inner_offset = outer_offset + 5*(rec_z_ind_b - k);
			GIJ = GI(m,k);

			txx[GIJ] += -c[m][k][1]*coe2*boundary_wavefields[inner_offset + 2*col_increment + vx_off];
			tzz[GIJ] += -c[m][k][2]*coe2*boundary_wavefields[inner_offset + 2*col_increment + vx_off];
			txz[GIJ] +=        -c19*coe2*boundary_wavefields[inner_offset + 1*col_increment + vz_off];
		}

		//On botmost cell only txx and tzz need to be adjusted
		k = rec_z_ind_b;
		inner_offset = outer_offset + 5*(rec_z_ind_b - k);
		GIJ = GI(m,k);
		txx[GIJ] += -c[m][k][1]*coe2*boundary_wavefields[inner_offset + 2*col_increment + vx_off];
		tzz[GIJ] += -c[m][k][2]*coe2*boundary_wavefields[inner_offset + 2*col_increment + vx_off];

		//On topmost cell only txz needs to be adjusted
		k = rec_z_ind_t;
		inner_offset = outer_offset + 5*(rec_z_ind_b - k);
		GIJ = GI(m,k);
		c19 = averag(c[m][k][3],c[m-1][k][3],c[m][k+1][3],c[m-1][k+1][3]);
		txz[GIJ] +=        -c19*coe2*boundary_wavefields[inner_offset + 1*col_increment + vz_off];

	//layer on boundary (mixed)
		m = rec_x_ind_l;
		outer_offset = side_offset + (m - (rec_x_ind_l-1))*col_increment;

		//general update
		for (k = rec_z_ind_b-1; k >= rec_z_ind_t+1; k--){
			c19 = averag(c[m][k][3],c[m-1][k][3],c[m][k+1][3],c[m-1][k+1][3]);
			inner_offset = outer_offset + 5*(rec_z_ind_b - k);
			GIJ = GI(m,k);

			txx[GIJ] += c[m][k][1]*(-coe1*boundary_wavefields[inner_offset - 0*col_increment + vx_off]
									-coe2*boundary_wavefields[inner_offset - 1*col_increment + vx_off]);
			tzz[GIJ] += c[m][k][2]*(-coe1*boundary_wavefields[inner_offset - 0*col_increment + vx_off]
									-coe2*boundary_wavefields[inner_offset - 1*col_increment + vx_off]);
			txz[GIJ] +=       -c19*( coe1*boundary_wavefields[inner_offset + 0*col_increment + vz_off]
									 coe2*boundary_wavefields[inner_offset + 1*col_increment + vz_off]);
		}

		//On botmost cell only txx and tzz need to be adjusted
		k = rec_z_ind_b;
		inner_offset = outer_offset + 5*(rec_z_ind_b - k);
		GIJ = GI(m,k);
		txx[GIJ] += c[m][k][1]*(-coe1*boundary_wavefields[inner_offset - 0*col_increment + vx_off]
								-coe2*boundary_wavefields[inner_offset - 1*col_increment + vx_off]);
		tzz[GIJ] += c[m][k][2]*(-coe1*boundary_wavefields[inner_offset - 0*col_increment + vx_off]
								-coe2*boundary_wavefields[inner_offset - 1*col_increment + vx_off]);

		//On topmost cell only txz needs to be adjusted
		k = rec_z_ind_t;
		inner_offset = outer_offset + 5*(rec_z_ind_b - k);
		GIJ = GI(m,k);
		c19 = averag(c[m][k][3],c[m-1][k][3],c[m][k+1][3],c[m-1][k+1][3]);
		txz[GIJ] +=       -c19*( coe1*boundary_wavefields[inner_offset + 0*col_increment + vz_off]
								 coe2*boundary_wavefields[inner_offset + 1*col_increment + vz_off]);

	//One layer right of boundary (perturbed fields)
		m = rec_x_ind_l+1;
		outer_offset = side_offset + (m - (rec_x_ind_l-1))*col_increment;
		//general update
		for (k = rec_z_ind_b-1; k >= rec_z_ind_t+1; k--){
			c19 = averag(c[m][k][3],c[m-1][k][3],c[m][k+1][3],c[m-1][k+1][3]);
			inner_offset = outer_offset + 5*(rec_z_ind_b - k);
			GIJ = GI(m,k);

			txx[GIJ] += c[m][k][1]*(-coe2*boundary_wavefields[inner_offset - 1*col_increment + vx_off]);
			tzz[GIJ] += c[m][k][2]*(-coe2*boundary_wavefields[inner_offset - 1*col_increment + vx_off]);
			txz[GIJ] +=        c19*(-coe2*boundary_wavefields[inner_offset - 2*col_increment + vz_off]);
		}
		//On botmost cell only txx and tzz need to be adjusted
		k = rec_z_ind_b;
		inner_offset = outer_offset + 5*(rec_z_ind_b - k);
		GIJ = GI(m,k);
		txx[GIJ] += c[m][k][1]*(-coe2*boundary_wavefields[inner_offset - 1*col_increment + vx_off]);
		tzz[GIJ] += c[m][k][2]*(-coe2*boundary_wavefields[inner_offset - 1*col_increment + vx_off]);

		//On topmost cell only txz needs to be adjusted
		k = rec_z_ind_t;
		inner_offset = outer_offset + 5*(rec_z_ind_b - k);
		GIJ = GI(m,k);
		c19 = averag(c[m][k][3],c[m-1][k][3],c[m][k+1][3],c[m-1][k+1][3]);
		txz[GIJ] +=        c19*(-coe2*boundary_wavefields[inner_offset - 2*col_increment + vz_off]);
}

//Using same offset variables as in fd2d_rec_wavefields. Somewhat redundant and will result in more lines of code, but easier to check if I am consistent.
void update_boundary_velocity_SSG(double* boundary_wavefields){
	unsigned long GIJ;
	int row_increment = 5*(rec_x_ind_r - rec_x_ind_l +1); //For every row we record on the top and bottom we store 5 quantities. tau_xx, tau_zz, tau_xz, vx and vz
	int col_increment = 5*(rec_z_ind_b - rec_z_ind_t +1);
	int time_offset = jj*(2*3*row_increment + 2*3*col_increment); //2 times 3 cols (left and right) 2 times 3 rows (top and bot). Using jj and not (jj-1) as in the stress update. For velocity update we use recorded stress at the same timestep loop iteration.
	int outer_offset, inner_offset;
	double c19;

	/*********************** TOP ***********************/
	/* ONLY DEAL WITH VERTICAL DERIVATIVE CORRECIONS   */
	int side_offset = time_offset; //counts total accumulated offset from previous sides and previous timesteps
	//One layer above boundary (scattered fields)
		k = rec_z_ind_t-1;
		outer_offset = side_offset + (k-(rec_z_ind_t-1))*row_increment;

		//general update
		for (m=rec_x_ind_l+1;m<=rec_x_ind_r-1;m++){
			inner_offset = outer_offset + 5*(m-rec_x_ind_l);
			GIJ = GI(m,k);
			vx[GIJ] += -coe2*2.0*dtdx/(c[m][k][0]+c[m-1][k][0])*boundary_wavefields[inner_offset + 1*row_increment + txz_off];
			vz[GIJ] += -coe2*2.0*dtdx/(c[m][k][0]+c[m][k+1][0])*boundary_wavefields[inner_offset + 2*row_increment + tzz_off];
		}

		//On leftmost cell only vz needs to be adjusted
		m = rec_x_ind_l;
		inner_offset = outer_offset + 5*(m-rec_x_ind_l);
		GIJ = GI(m,k);
		vz[GIJ] += -coe2*2.0*dtdx/(c[m][k][0]+c[m][k+1][0])*boundary_wavefields[inner_offset + 2*row_increment + tzz_off];

		//On rightmost cell only vx needs to be adjusted
		m = rec_x_ind_r;
		inner_offset = outer_offset + 5*(m-rec_x_ind_l);
		GIJ = GI(m,k);
		vx[GIJ] += -coe2*2.0*dtdx/(c[m][k][0]+c[m-1][k][0])*boundary_wavefields[inner_offset + 1*row_increment + txz_off];

	//layer on boundary (mixed). Horizontal derivative contributions will be updated by left and right boundary pass
		k = rec_z_ind_t;
		outer_offset = side_offset + (k-(rec_z_ind_t-1))*row_increment;

		//general update
		for (m=rec_x_ind_l+1;m<=rec_x_ind_r-1;m++){
			inner_offset = outer_offset + 5*(m-rec_x_ind_l);
			GIJ = GI(m,k);
			vx[GIJ] += -2.0*dtdx/(c[m][k][0]+c[m-1][k][0])*( coe1*boundary_wavefields[inner_offset + 0*row_increment + txz_off]
															+coe2*boundary_wavefields[inner_offset + 1*row_increment + txz_off]);
			vz[GIJ] +=  2.0*dtdx/(c[m][k][0]+c[m][k+1][0])*(-coe1*boundary_wavefields[inner_offset - 0*row_increment + tzz_off]
															-coe2*boundary_wavefields[inner_offset - 1*row_increment + tzz_off]);

			//printf("+0 txz SHOULD BE SAME: %e, %e \n",boundary_wavefields[inner_offset + 0*row_increment + txz_off],txz[GIJ]);
		}

		//On leftmost cell only vz needs to be adjusted
		m = rec_x_ind_l;
		inner_offset = outer_offset + 5*(m-rec_x_ind_l);
		GIJ = GI(m,k);
		vz[GIJ] +=  2.0*dtdx/(c[m][k][0]+c[m][k+1][0])*(-coe1*boundary_wavefields[inner_offset - 0*row_increment + tzz_off]
														-coe2*boundary_wavefields[inner_offset - 1*row_increment + tzz_off]);

		//On rightmost cell only vx needs to be adjusted
		m = rec_x_ind_r;
		inner_offset = outer_offset + 5*(m-rec_x_ind_l);
		GIJ = GI(m,k);
		vx[GIJ] += -2.0*dtdx/(c[m][k][0]+c[m-1][k][0])*( coe1*boundary_wavefields[inner_offset + 0*row_increment + txz_off]
														+coe2*boundary_wavefields[inner_offset + 1*row_increment + txz_off]);

	//One layer below boundary (perturbed fields). Horizontal derivative contributions will be updated by left and right boundary pass
		k = rec_z_ind_t+1;
		outer_offset = side_offset + (k-(rec_z_ind_t-1))*row_increment;

		//general update
		for (m=rec_x_ind_l+1;m<=rec_x_ind_r-1;m++){
			inner_offset = outer_offset + 5*(m-rec_x_ind_l);
			GIJ = GI(m,k);

			vx[GIJ] += 2.0*dtdx/(c[m][k][0]+c[m-1][k][0])*(-coe2*boundary_wavefields[inner_offset - 2*row_increment + txz_off]);
			vz[GIJ] += 2.0*dtdx/(c[m][k][0]+c[m][k+1][0])*(-coe2*boundary_wavefields[inner_offset - 1*row_increment + tzz_off]);

			//printf("txz +1 SHOULD BE SAME: %e, %e \n",boundary_wavefields[inner_offset + 0*row_increment + txz_off],txz[GIJ]);
			//printf("txx +1 SHOULD BE SAME: %e, %e \n",boundary_wavefields[inner_offset + 0*row_increment + txx_off],txx[GIJ]);
			//printf("tzz +1 SHOULD BE SAME: %e, %e \n",boundary_wavefields[inner_offset + 0*row_increment + tzz_off],tzz[GIJ]);
		}

		//On leftmost cell only vz needs to be adjusted
		m = rec_x_ind_l;
		inner_offset = outer_offset + 5*(m-rec_x_ind_l);
		GIJ = GI(m,k);
		vz[GIJ] += 2.0*dtdx/(c[m][k][0]+c[m][k+1][0])*(-coe2*boundary_wavefields[inner_offset - 1*row_increment + tzz_off]);

		//On rightmost cell only vx needs to be adjusted
		m = rec_x_ind_r;
		inner_offset = outer_offset + 5*(m-rec_x_ind_l);
		GIJ = GI(m,k);
		vx[GIJ] += 2.0*dtdx/(c[m][k][0]+c[m-1][k][0])*(-coe2*boundary_wavefields[inner_offset - 2*row_increment + txz_off]);

	/*********************** RIGHT ***********************/
	/* ONLY DEAL WITH HORIZONTAL DERIVATIVE CORRECIONS   */
	side_offset += 3*row_increment; //counts total accumulated offset from previous sides and previous timesteps
	//One layer right of boundary (scattered fields)
		m = rec_x_ind_r+1;
		outer_offset = side_offset + ((rec_x_ind_r+1)-m)*col_increment;

		//general update
		for (k=rec_z_ind_t+1;k<=rec_z_ind_b-1;k++){
			inner_offset = outer_offset + 5*(k-rec_z_ind_t);
			GIJ = GI(m,k);
			vx[GIJ] +=  coe2*2.0*dtdx/(c[m][k][0]+c[m-1][k][0])*boundary_wavefields[inner_offset + 2*col_increment + txx_off];
			vz[GIJ] +=  coe2*2.0*dtdx/(c[m][k][0]+c[m][k+1][0])*boundary_wavefields[inner_offset + 1*col_increment + txz_off];
		}

		//On topmost cell only vz needs to be adjusted
		k = rec_z_ind_t;
		inner_offset = outer_offset + 5*(k-rec_z_ind_t);
		GIJ = GI(m,k);
		vz[GIJ] +=  coe2*2.0*dtdx/(c[m][k][0]+c[m][k+1][0])*boundary_wavefields[inner_offset + 1*col_increment + txz_off];

		//On botmost cell only vx needs to be adjusted
		k = rec_z_ind_b;
		inner_offset = outer_offset + 5*(k-rec_z_ind_t);
		GIJ = GI(m,k);
		vx[GIJ] +=  coe2*2.0*dtdx/(c[m][k][0]+c[m-1][k][0])*boundary_wavefields[inner_offset + 2*col_increment + txx_off];

	//layer on boundary (mixed). Vertical updates will come from top and bot pass
		m = rec_x_ind_r;
		outer_offset = side_offset + ((rec_x_ind_r+1)-m)*col_increment;

		//general update
		for (k=rec_z_ind_t+1;k<=rec_z_ind_b-1;k++){
			inner_offset = outer_offset + 5*(k-rec_z_ind_t);
			GIJ = GI(m,k);
			vx[GIJ] +=  2.0*dtdx/(c[m][k][0]+c[m-1][k][0])*( coe1*boundary_wavefields[inner_offset - 0*col_increment + txx_off]
															+coe2*boundary_wavefields[inner_offset - 1*col_increment + txx_off]);
			vz[GIJ] +=  2.0*dtdx/(c[m][k][0]+c[m][k+1][0])*( coe1*boundary_wavefields[inner_offset + 0*col_increment + txz_off]
															+coe2*boundary_wavefields[inner_offset + 1*col_increment + txz_off]);
		}

		//On topmost cell only vz needs to be adjusted
		k = rec_z_ind_t;
		inner_offset = outer_offset + 5*(k-rec_z_ind_t);
		GIJ = GI(m,k);
		vz[GIJ] +=  2.0*dtdx/(c[m][k][0]+c[m][k+1][0])*( coe1*boundary_wavefields[inner_offset + 0*col_increment + txz_off]
														+coe2*boundary_wavefields[inner_offset + 1*col_increment + txz_off]);

		//On botmost cell only vx needs to be adjusted
		k = rec_z_ind_b;
		inner_offset = outer_offset + 5*(k-rec_z_ind_t);
		GIJ = GI(m,k);
		vx[GIJ] +=  2.0*dtdx/(c[m][k][0]+c[m-1][k][0])*( coe1*boundary_wavefields[inner_offset - 0*col_increment + txx_off]
														+coe2*boundary_wavefields[inner_offset - 1*col_increment + txx_off]);

	//One layer left of boundary  (perturbed fields)
		m = rec_x_ind_r-1;
		outer_offset = side_offset + ((rec_x_ind_r+1)-m)*col_increment;

		//general update
		for (k=rec_z_ind_t+1;k<=rec_z_ind_b-1;k++){
			inner_offset = outer_offset + 5*(k-rec_z_ind_t);
			GIJ = GI(m,k);
			vx[GIJ] +=  2.0*dtdx/(c[m][k][0]+c[m-1][k][0])*coe2*boundary_wavefields[inner_offset - 1*col_increment + txx_off];
			vz[GIJ] +=  2.0*dtdx/(c[m][k][0]+c[m][k+1][0])*coe2*boundary_wavefields[inner_offset - 2*col_increment + txz_off];
		}

		//On topmost cell only vz needs to be adjusted
		k = rec_z_ind_t;
		inner_offset = outer_offset + 5*(k-rec_z_ind_t);
		GIJ = GI(m,k);
		vz[GIJ] +=  2.0*dtdx/(c[m][k][0]+c[m][k+1][0])*coe2*boundary_wavefields[inner_offset - 2*col_increment + txz_off];

		//On botmost cell only vx needs to be adjusted
		k = rec_z_ind_b;
		inner_offset = outer_offset + 5*(k-rec_z_ind_t);
		GIJ = GI(m,k);
		vx[GIJ] +=  2.0*dtdx/(c[m][k][0]+c[m-1][k][0])*coe2*boundary_wavefields[inner_offset - 1*col_increment + txx_off];

	/********************** BOT ************************/
	/* ONLY DEAL WITH VERTICAL DERIVATIVE CORRECIONS   */
	side_offset += 3*col_increment; //counts total accumulated offset from previous sides and previous timesteps
	//One layer below bottom boundary (scattered fields)
		k = rec_z_ind_b+1;
		outer_offset = side_offset + ((rec_z_ind_b+1)-k)*row_increment;
		for (m=rec_x_ind_r-1; m>=rec_x_ind_l+1; m--){
			inner_offset = outer_offset + 5*(rec_x_ind_r - m);
			GIJ = GI(m,k);
			vx[GIJ] += coe2*2.0*dtdx/(c[m][k][0]+c[m-1][k][0])*boundary_wavefields[inner_offset + 2*row_increment + txz_off];
			vz[GIJ] += coe2*2.0*dtdx/(c[m][k][0]+c[m][k+1][0])*boundary_wavefields[inner_offset + 1*row_increment + tzz_off];
		}

		//On rightmost cell only vx needs to be adjusted
		m=rec_x_ind_r;
		GIJ = GI(m,k);
		inner_offset = outer_offset + 5*(rec_x_ind_r - m);
		vx[GIJ] += coe2*2.0*dtdx/(c[m][k][0]+c[m-1][k][0])*boundary_wavefields[inner_offset + 2*row_increment + txz_off];

		//On leftmost cell only vz needs to be adjusted
		m=rec_x_ind_l;
		GIJ = GI(m,k);
		inner_offset = outer_offset + 5*(rec_x_ind_r - m);
		vz[GIJ] += coe2*2.0*dtdx/(c[m][k][0]+c[m][k+1][0])*boundary_wavefields[inner_offset + 1*row_increment + tzz_off];

	//layer on boundary (mixed). Horizontal updates will come from left and right pass
		k = rec_z_ind_b;
		outer_offset = side_offset + ((rec_z_ind_b+1)-k)*row_increment;
		for (m=rec_x_ind_r-1; m>=rec_x_ind_l+1; m--){
			inner_offset = outer_offset + 5*(rec_x_ind_r - m);
			GIJ = GI(m,k);
			vx[GIJ] += 2.0*dtdx/(c[m][k][0]+c[m-1][k][0])*( coe1*boundary_wavefields[inner_offset - 0*row_increment + txz_off]
														   +coe2*boundary_wavefields[inner_offset - 1*row_increment + txz_off]);
			vz[GIJ] += 2.0*dtdx/(c[m][k][0]+c[m][k+1][0])*( coe1*boundary_wavefields[inner_offset + 0*row_increment + tzz_off]
														   +coe2*boundary_wavefields[inner_offset + 1*row_increment + tzz_off]);
		}
		//On rightmost cell only vx needs to be adjusted
		m=rec_x_ind_r;
		GIJ = GI(m,k);
		inner_offset = outer_offset + 5*(rec_x_ind_r - m);
		vx[GIJ] += 2.0*dtdx/(c[m][k][0]+c[m-1][k][0])*( coe1*boundary_wavefields[inner_offset - 0*row_increment + txz_off]
													   +coe2*boundary_wavefields[inner_offset - 1*row_increment + txz_off]);

		//On leftmost cell only vz needs to be adjusted
		m=rec_x_ind_l;
		GIJ = GI(m,k);
		inner_offset = outer_offset + 5*(rec_x_ind_r - m);
		vz[GIJ] += 2.0*dtdx/(c[m][k][0]+c[m][k+1][0])*( coe1*boundary_wavefields[inner_offset + 0*row_increment + tzz_off]
													   +coe2*boundary_wavefields[inner_offset + 1*row_increment + tzz_off]);

	//layer above boundary (perturbed). Horizontal updates will come from left and right pass
		k = rec_z_ind_b-1;
		outer_offset = side_offset + ((rec_z_ind_b+1)-k)*row_increment;
		for (m=rec_x_ind_r-1; m>=rec_x_ind_l+1; m--){
			inner_offset = outer_offset + 5*(rec_x_ind_r - m);
			GIJ = GI(m,k);
			vx[GIJ] += 2.0*dtdx/(c[m][k][0]+c[m-1][k][0])*coe2*boundary_wavefields[inner_offset - 1*row_increment + txz_off];
			vz[GIJ] += 2.0*dtdx/(c[m][k][0]+c[m][k+1][0])*coe2*boundary_wavefields[inner_offset - 2*row_increment + tzz_off];
		}

		//On rightmost cell only vx needs to be adjusted
		m=rec_x_ind_r;
		GIJ = GI(m,k);
		inner_offset = outer_offset + 5*(rec_x_ind_r - m);
		vx[GIJ] += 2.0*dtdx/(c[m][k][0]+c[m-1][k][0])*coe2*boundary_wavefields[inner_offset - 1*row_increment + txz_off];

		//On leftmost cell only vz needs to be adjusted
		m=rec_x_ind_l;
		GIJ = GI(m,k);
		inner_offset = outer_offset + 5*(rec_x_ind_r - m);
		vz[GIJ] += 2.0*dtdx/(c[m][k][0]+c[m][k+1][0])*coe2*boundary_wavefields[inner_offset - 2*row_increment + tzz_off];

	/*********************** LEFT ************************/
	/* ONLY DEAL WITH HORIZONTAL DERIVATIVE CORRECIONS   */
	side_offset += 3*row_increment; //counts total accumulated offset from previous sides and previous timesteps

	//One layer left of boundary (scattered fields)
		m = rec_x_ind_l-1;
		outer_offset = side_offset + (m - (rec_x_ind_l-1))*col_increment;

		//general update
		for (k = rec_z_ind_b-1; k >= rec_z_ind_t+1; k--){
			inner_offset = outer_offset + 5*(rec_z_ind_b - k);
			GIJ = GI(m,k);

			vx[GIJ] +=  -coe2*2.0*dtdx/(c[m][k][0]+c[m-1][k][0])*boundary_wavefields[inner_offset + 1*col_increment + txx_off];
			vz[GIJ] +=  -coe2*2.0*dtdx/(c[m][k][0]+c[m][k+1][0])*boundary_wavefields[inner_offset + 2*col_increment + txz_off];
		}

		//On botmost cell only vx needs to be adjusted
		k = rec_z_ind_b;
		GIJ = GI(m,k);
		inner_offset = outer_offset + 5*(rec_z_ind_b - k);
		vx[GIJ] +=  -coe2*2.0*dtdx/(c[m][k][0]+c[m-1][k][0])*boundary_wavefields[inner_offset + 1*col_increment + txx_off];

		//On topmost cell only vz needs to be adjusted
		k = rec_z_ind_t;
		GIJ = GI(m,k);
		inner_offset = outer_offset + 5*(rec_z_ind_b - k);
		vz[GIJ] +=  -coe2*2.0*dtdx/(c[m][k][0]+c[m][k+1][0])*boundary_wavefields[inner_offset + 2*col_increment + txz_off];

	//layer on boundary (mixed). Vertical updates will come from top and bot pass
		m = rec_x_ind_l;
		outer_offset = side_offset + (m - (rec_x_ind_l-1))*col_increment;

		//general update
		for (k = rec_z_ind_b-1; k >= rec_z_ind_t+1; k--){
			inner_offset = outer_offset + 5*(rec_z_ind_b - k);
			GIJ = GI(m,k);

			vx[GIJ] +=   -2.0*dtdx/(c[m][k][0]+c[m-1][k][0])*( coe1*boundary_wavefields[inner_offset + 0*col_increment + txx_off]
															  +coe2*boundary_wavefields[inner_offset + 1*col_increment + txx_off]);
			vz[GIJ] +=    2.0*dtdx/(c[m][k][0]+c[m][k+1][0])*(-coe1*boundary_wavefields[inner_offset - 0*col_increment + txz_off]
															  -coe2*boundary_wavefields[inner_offset - 1*col_increment + txz_off]);
		}

		//On botmost cell only vx needs to be adjusted
		k = rec_z_ind_b;
		GIJ = GI(m,k);
		inner_offset = outer_offset + 5*(rec_z_ind_b - k);
		vx[GIJ] +=   -2.0*dtdx/(c[m][k][0]+c[m-1][k][0])*( coe1*boundary_wavefields[inner_offset + 0*col_increment + txx_off]
														  +coe2*boundary_wavefields[inner_offset + 1*col_increment + txx_off]);

		//On topmost cell only vz needs to be adjusted
		k = rec_z_ind_t;
		GIJ = GI(m,k);
		inner_offset = outer_offset + 5*(rec_z_ind_b - k);
		vz[GIJ] +=    2.0*dtdx/(c[m][k][0]+c[m][k+1][0])*(-coe1*boundary_wavefields[inner_offset - 0*col_increment + txz_off]
														  -coe2*boundary_wavefields[inner_offset - 1*col_increment + txz_off]);

	//One layer right of boundary  (perturbed fields). Vertical updates will come from top and bot pass
		m = rec_x_ind_l+1;
		outer_offset = side_offset + (m - (rec_x_ind_l-1))*col_increment;

		//general update
		for (k = rec_z_ind_b-1; k >= rec_z_ind_t+1; k--){
			inner_offset = outer_offset + 5*(rec_z_ind_b - k);
			GIJ = GI(m,k);

			vx[GIJ] +=  2.0*dtdx/(c[m][k][0]+c[m-1][k][0])*(-coe2)*boundary_wavefields[inner_offset - 2*col_increment + txx_off];
			vz[GIJ] +=  2.0*dtdx/(c[m][k][0]+c[m][k+1][0])*(-coe2)*boundary_wavefields[inner_offset - 1*col_increment + txz_off];
		}

		//On botmost cell only vx needs to be adjusted
		k = rec_z_ind_b;
		GIJ = GI(m,k);
		inner_offset = outer_offset + 5*(rec_z_ind_b - k);
		vx[GIJ] +=  2.0*dtdx/(c[m][k][0]+c[m-1][k][0])*(-coe2)*boundary_wavefields[inner_offset - 2*col_increment + txx_off];

		//On topmost cell only vz needs to be adjusted
		k = rec_z_ind_t;
		GIJ = GI(m,k);
		inner_offset = outer_offset + 5*(rec_z_ind_b - k);
		vz[GIJ] +=  2.0*dtdx/(c[m][k][0]+c[m][k+1][0])*(-coe2)*boundary_wavefields[inner_offset - 1*col_increment + txz_off];
}
