//Bram Willemsen, feb 2016
//Here I will record the wavefield at the boundary of the truncated domain, which will be used for local solves later on

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "fd2d.h"

void rec_wavefields(double* boundary_wavefields){
	//The way I implement the time-offset is more efficient from a memory access point of view
	//But downside is that it is harder to plot the boundary traces, as trace entries for a specific quantity are not next to each other in memory.
	//When using the local solver we basically will access the boundary wavefield one timestep at a time.
	//So for use it is efficient to put all quantities for the same timestep next to each other in memory
	if (jj==0){
		printf("Use OMP when recording wavefields? Since only boundary perhaps too little work and only overhead? Since it involves copying, bandwidth limited ?\n");
	}
	unsigned long GIJ;
	int row_increment = 5*(rec_x_ind_r - rec_x_ind_l +1); //For every row we record on the top and bottom we store 5 quantities. tau_xx, tau_zz, tau_xz, vx and vz
	int col_increment = 5*(rec_z_ind_b - rec_z_ind_t +1);
	int time_offset = jj*(2*3*row_increment + 2*3*col_increment); //2 times 3 cols (left and right) 2 times 3 rows (top and bot).

	//TOP (LEFT TO RIGHT)
	int side_offset = time_offset; //counts total accumulated offset from previous sides and previous timesteps
	int outer_offset, inner_offset;
	for(k=rec_z_ind_t-1;k<=rec_z_ind_t+1; k++){ //Three rows on top, centered around the boundary
		outer_offset = side_offset + (k-(rec_z_ind_t-1))*row_increment;
		for(m=rec_x_ind_l;m<=rec_x_ind_r; m++){ //all x nodes following the recording surface
			inner_offset = outer_offset + 5*(m-rec_x_ind_l);
			GIJ=GI(m,k);

			boundary_wavefields[inner_offset + txx_off] = txx[GIJ];
			boundary_wavefields[inner_offset + tzz_off] = tzz[GIJ];
			boundary_wavefields[inner_offset + txz_off] = txz[GIJ];
			boundary_wavefields[inner_offset +  vx_off] =  vx[GIJ];
			boundary_wavefields[inner_offset +  vz_off] =  vz[GIJ];
		}
	}

	//RIGHT (TOP TO BOT)
	side_offset += 3*row_increment; //3 rows, each entry in row we record 5 vars
	for (m=rec_x_ind_r+1; m>=rec_x_ind_r-1;m--){
		outer_offset = side_offset + ((rec_x_ind_r+1)-m)*col_increment;
		for(k=rec_z_ind_t;k<=rec_z_ind_b;k++){
			inner_offset = outer_offset + 5*(k-rec_z_ind_t);
			GIJ=GI(m,k);

			boundary_wavefields[inner_offset + txx_off] = txx[GIJ];
			boundary_wavefields[inner_offset + tzz_off] = tzz[GIJ];
			boundary_wavefields[inner_offset + txz_off] = txz[GIJ];
			boundary_wavefields[inner_offset +  vx_off] =  vx[GIJ];
			boundary_wavefields[inner_offset +  vz_off] =  vz[GIJ];
		}
	}

	//BOT (RIGHT TO LEFT)
	side_offset += 3*col_increment;
	for (k=rec_z_ind_b+1; k>= rec_z_ind_b-1; k--){
		outer_offset = side_offset + ((rec_z_ind_b+1)-k)*row_increment;
		for (m=rec_x_ind_r; m>=rec_x_ind_l; m--){
			inner_offset = outer_offset + 5*(rec_x_ind_r - m);
			GIJ=GI(m,k);

			boundary_wavefields[inner_offset + txx_off] = txx[GIJ];
			boundary_wavefields[inner_offset + tzz_off] = tzz[GIJ];
			boundary_wavefields[inner_offset + txz_off] = txz[GIJ];
			boundary_wavefields[inner_offset +  vx_off] =  vx[GIJ];
			boundary_wavefields[inner_offset +  vz_off] =  vz[GIJ];
		}
	}

	//LEFT (BOT TO TOP)
	side_offset += 3*row_increment;
	for (m=rec_x_ind_l-1; m <= rec_x_ind_l+1; m++){
		outer_offset = side_offset + (m - (rec_x_ind_l-1))*col_increment;
		for (k = rec_z_ind_b; k >= rec_z_ind_t; k--){
			inner_offset = outer_offset + 5*(rec_z_ind_b - k);
			GIJ=GI(m,k);

			boundary_wavefields[inner_offset + txx_off] = txx[GIJ];
			boundary_wavefields[inner_offset + tzz_off] = tzz[GIJ];
			boundary_wavefields[inner_offset + txz_off] = txz[GIJ];
			boundary_wavefields[inner_offset +  vx_off] =  vx[GIJ];
			boundary_wavefields[inner_offset +  vz_off] =  vz[GIJ];
		}
	}
}
