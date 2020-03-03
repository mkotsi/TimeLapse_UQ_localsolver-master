/*=================================================================
  * fd2d_readin_struct.c -- readin model parameters from input struct
  *
  * 2D finite difference modeling seismic waves propagation in fractured media
  *
  *  Xinding Fang, MIT-ERL, May 2009
  *  Email: xinfang@mit.edu
  *
  *  This particular function is written by Bram Willemsen
  *  MIT-ERL, 2016
  *  Email: bramwillemsen@gmail.com
 =================================================================*/

/***********************************************************
 *
 *  units:    density    kg/m3
 *            velocity   m/s
 *            dt         sec
 *            dr         m
 *            frequency  Hz
 *
 ***********************************************************/
#include <omp.h>
//#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "fd2d.h"
#include "funlist.h"

int readin_from_info_struct(struct info_struct *istruct)
{
	int li;

	MM = istruct->MM;
	KK = istruct->KK;
	nabs = istruct->nabs;
	ifsbc = istruct->ifsbc;
	d0factor = istruct->d0factor;
	PPW0 = istruct->PPW0;
	p_power = istruct->p_power;
	dx = istruct->dx;
	dt = istruct->dt;
	itimestep = istruct->itimestep;
	gx0 = istruct->gx0;
	gz0 = istruct->gz0;
	iwavelet = istruct->iwavelet; //right now just support 0, but later will allow 2 as well (but don't read from file, pass wavelet directly from pysit)
	amp0 = istruct->amp0;
	freq0 = istruct->freq0;
	isourcecomp = istruct->isourcecomp;
	nsrc  = istruct->nsrc;
	srcxz = d2matrix(2,nsrc);
	srcxz[0] = istruct->srcx;
	srcxz[1] = istruct->srcz;
	screen = istruct->screen;
	snap = istruct->snap;
	hist = istruct->hist;
	ss = istruct->ss;
	bparams = istruct->bparams;
	nhist = istruct->nhist;
	rcvxz    = d2matrix(2,nhist);
	rcvxz[0] = istruct->rcvx;
	rcvxz[1] = istruct->rcvz;
	traces_mem = istruct->traces_mem;
	snaps_mem = istruct->snaps_mem;
	traces_output_dir = istruct->traces_output_dir;
	snaps_output_dir = istruct->snaps_output_dir;

	tfin = itimestep*dt;     /* itimestep = number of time steps */
	dtdx = dt/dx;	// used for updating velocity

	for(k=0;k<NOUTVAR;k++) { //copy array content
		histvar[k] = istruct->histvar[k];
	}

	/////////////////////COPIED////////////////////
    if(ifsbc==0) nabs_top=nabs; // no free surface, PML above the model
    else         nabs_top=0;    // free surface

    /* update everything */
    kk = KK + nabs_top + nabs;
    mm = MM + 2*nabs;
    nelm = mm*kk-MM*KK;


	gx = d1matrix(mm);  /* gx, gz are pointers */
	gz = d1matrix(kk);
	for(m=0;m<MM;m++)    gx[nabs+m]=gx0 + m*dx;
	for(m=0;m<nabs;m++) {
	     gx[m]=gx[nabs]-(nabs-m)*dx;
	     gx[nabs+MM+m]=gx[nabs+MM-1]+(m+1)*dx;
	}
	for(k=0;k<KK;k++)           gz[nabs_top+k]=gz0+k*dx;
	for(k=0;k<nabs_top;k++)     gz[k]=gz[nabs_top]-(nabs_top-k)*dx;
	for(k=KK+nabs_top;k<kk;k++) gz[k]=gz[KK+nabs_top-1]+(k-KK-nabs_top+1)*dx;

	//////////////////////// SETUP SOURCE INDEX ////////////////////////
	srcind  = i2matrix(2,nsrc);  /* source location index */

	if (!bparams.local_solve){ //Normal operation mode
		for (j=0;j<nsrc;j++) { //copied from the old fd2d_readin.c
			if ( srcxz[0][j] < gx[0] || srcxz[0][j] > gx[mm-1] ||
				 srcxz[1][j] < gz[0] || srcxz[1][j] > gz[kk-1]    ) {
					 printf("***Error: source position %i is not on grid\n", j);
					 printf("    srcxz = [%g, %g] \n",srcxz[0][j],srcxz[1][j]);
					 printf("***Error: source position %i is not on grid\n", j+1);
					 printf("    srcxz = [%g,  %g] \n",srcxz[0][j],srcxz[1][j]);
					 return(8);
			}

			for (m=0;m<mm;m++) {
				if (fabs(gx[m]-srcxz[0][j])<=0.5*dx) { printf("m: %i, gx: %e, src: %e\n",m,gx[m],srcxz[0][j]); srcind[0][j] = m; break;}
			}
			for (k=0;k<kk;k++) {
				if (fabs(gz[k]-srcxz[1][j])<=0.5*dx) { printf("k: %i, gz: %e, src: %e\n",k,gz[k],srcxz[1][j]); srcind[1][j] = k;  break; }
			}

			if ( srcind[0][j] < 0 || srcind[0][j] >= mm ||
				 srcind[1][j] < 0 || srcind[1][j] >= kk ) {
					printf("Error: can't find source position %i on grid\n",j+1);
					printf("  srcxz  = [%g, %g] \n",srcxz[0][j],srcxz[1][j]);
					printf("  srcind = [%i, %i] \n",srcind[0][j],srcind[1][j]);
					printf("Error: can't find source position %i on grid\n",j);
					return(8);
			}
		}
	}
	else{ //Local solve mode. Right now I am just using a dummy source with 0 amplitude so I don't have to change logic in the rest of the code. Source will be introduced through boundary condition. Assuming source external to local domain. Not really clean code
		amp0 = 0.0;
		for (j=0;j<nsrc;j++) {
			srcind[0][j] = nabs;
			srcind[1][j] = nabs;
		}
	}
    //////////////////////// END SETUP SOURCE INDEX ////////////////////////

    //////////////////////// SETUP RECEIVER INDEX //////////////////////////



    if ( hist.l ) { //Copy from fd2d_readin.c
		rcvval = f1matrix(nhist);
		rcvind = i2matrix(2,nhist);
    	//if (!bparams.local_solve){ //Normal operation mode
    		if(nhist<=0) {
    			printf("***Error: Bad number of receiver points.\n");
    			return(6);
    		}

    		for(li=0;li<nhist;li++) {
    			if ( rcvxz[0][li] < gx[0] || rcvxz[0][li] > gx[mm-1] ||
    				 rcvxz[1][li] < gz[0] || rcvxz[1][li] > gz[kk-1] ) {
    					printf("***Error: receiver %i is outside grid\n",li);
    					printf("     %7.2f %7.2f \n", rcvxz[0][li],rcvxz[1][li]);
    					printf("   grid x limits:   %f %f \n", gx[0], gx[mm-1]);
    					printf("   grid z limits:   %f %f \n", gz[0], gz[kk-1]);
    					return(7);
    			}

    			rcvind[0][li] = 0;
    			rcvind[1][li] = 0;

    			for (m=0;m<mm;m++) {
    				if ( fabs(gx[m] - rcvxz[0][li]) <= .5*dx ) { rcvind[0][li] = m; break;}
    			}
    			for (k=0;k<kk;k++) {
    				if ( fabs(gz[k] - rcvxz[1][li]) <= .5*dx ) { rcvind[1][li] = k; break;}
    			}

    			if ( rcvind[0][li] < 0 || rcvind[1][li] < 0 ) {
    				printf("***Error: can't find rcvr %i position on grid\n",li);
    				printf("  rcvind = [%i, %i]\n",rcvind[0][li],rcvind[1][li]);
    				printf("  rcvxz  = [%g, %g]\n",rcvxz[0][li],rcvxz[1][li]);
    				return(7);
    			}

    		}/* for(li=0;li<nhist;li++) */

    		printf(" receiver coords:\n");
    		printf(" -------- -------\n");
    		printf("  number of receivers = %i\n",nhist);
    		printf("  number of receivers = %i\n",nhist);
    		printf("   Receiver number    x   (index)       z   (index)\n");
    		printf("    ----------       -------------     --------------    \n");

    		for(li=0;li<nhist;li++)
    			printf("  %3i %7.4f (%4i=%7.4f m) %7.4f (%4i=%7.4f m) \n",li+1,
    					rcvxz[0][li],rcvind[0][li],gx[rcvind[0][li]],
						rcvxz[1][li],rcvind[1][li],gz[rcvind[1][li]]);
    		printf("\n");
    	//}
    	/*else{ //propagate wavefield to receivers in local solver
    		//For now I just don't do anything. So all receivers are at pixel 0
    		for (j=0;j<nhist;j++) {
    			rcvind[0][j] = nabs;
    			rcvind[1][j] = nabs;
    		}
    	}*/
    }/* if(hist.l) */

    //////////////////////// END SETUP RECEIVER INDEX //////////////////////

    //////////////////////// SETUP LOCAL SOLVER NODE NUMBERS ///////////////
    // When we do a local solve, have gx0 and gz0 correspond to full grid coordinates
    // and have mm, kk, MM and KK give the size of the submesh.
    int found_rec_x_ind_l =0, found_rec_x_ind_r =0, found_rec_z_ind_t =0, found_rec_z_ind_b =0;

    if (bparams.rec_boundary || bparams.local_solve){ //we need both for recording wavefields and when we want to do local solve
        for (m=0;m<mm;m++) { //find x nodes.
          if (fabs(gx[m]-bparams.rec_x_l )<=0.001*dx) { rec_x_ind_l  = m; found_rec_x_ind_l = 1;}
          if (fabs(gx[m]-bparams.rec_x_r )<=0.001*dx) { rec_x_ind_r  = m; found_rec_x_ind_r = 1;}
        }

        for (k=0;k<kk;k++) { //find z nodes.
          if (fabs(gz[k]-bparams.rec_z_t )<=0.001*dx) { rec_z_ind_t  = k; found_rec_z_ind_t = 1;}
          if (fabs(gz[k]-bparams.rec_z_b )<=0.001*dx) { rec_z_ind_b  = k; found_rec_z_ind_b = 1;}
        }

        if (!found_rec_x_ind_l || !found_rec_x_ind_r ||!found_rec_z_ind_t ||!found_rec_z_ind_b){
        	printf("***Error: Could not find all node numbers for truncation boundary\n");
        	return(9);
        }
        printf("REC NODES: %i, %i, %i, %i \n", rec_x_ind_l, rec_x_ind_r, rec_z_ind_t, rec_z_ind_b);

        //sanity checks
        if (bparams.rec_x_l >= bparams.rec_x_r || bparams.rec_z_t >= bparams.rec_z_b){
        	printf("***Error: Invalid truncation square geometry\n");
        }
    }

    //////////////////////// END SETUP LOCAL SOLVER NODE NUMBERS ///////////



	//for(int k=0;k<nsrc;k++) { //print array stuff
	//	printf("Source x position: %f vs %f\n",istruct->srcx[k], srcxz[0][k]);
	//	printf("Source z position: %f vs %f\n",istruct->srcz[k], srcxz[1][k]);
	//}

	printf("hist: l=%i, ii=%i, ie=%i, di=%i\n",hist.l,hist.ii,hist.ie,hist.di);


    /////////////////////END COPIED////////////////

	//hardcoded
	iGRID = 0; // For now we work with standard grid
	modeltype = 2; //new case I made, will get from input ndarrays
	NDF = 0; //no fractures
	iHeter = 0; //no heterogeneous layers

	return 0;
}
