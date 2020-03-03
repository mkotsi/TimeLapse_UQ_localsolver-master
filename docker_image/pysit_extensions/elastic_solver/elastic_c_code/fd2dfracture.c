/*=================================================================
  * fd2dfracture.c -- main function
  *
  * Time Domain Finite Difference in 2-D Cartesian Coord's
  *    this code can use 3 different staggered FD schemes (staggered in space & time):
  *      (i)   Standard Staggered Grid with 4th-order in space and 2nd-order in time accuracy
  *      (ii)  2nd-order Rotated Staggered Grid with 2nd-order in space and 2nd-order in time accuracy
  *      (iii) 4th-order Rotated Staggered Grid with 4th-order in space and 2nd-order in time accuracy
  *
  * finite difference modeling of seismic waves propagation in fractured media (P-SV)
  * 
  *  Xinding Fang, MIT-ERL
  *  Email: xinfang@mit.edu
  *  Last update: 2010-10-5
  *-----------------------------------------------------------------------------------------------
  * Log:
      2010-9-28: implement Wei Zhang's unsplit complex frequency-shifted PML, Xinding & Xuefeng
  *-----------------------------------------------------------------------------------------------
  *
  * Note of Xinding:
       do elastic constants averaging in SSG

  * Bram, partially integrated the solver into pysit for the standard staggered grid.
  * Also some functionality to record wavefield along boundary of truncated domain
  * This can then later be used to do local solves.
 =================================================================*/
#include <omp.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "fd2d.h"
#include "funlist.h"

//depending on the way we use the solver, some pointers will not be used (will just point to an array of length 0)
//Perhaps not the best way to implement this. Should have used some kind structure with to contain all output
int solve(struct info_struct *istruct, const double * vp, const double * rho, const double * vs, double* wavefield_mem, double* wavefield_times, double* shotgather_mem, double* shotgather_times, double* boundary_wavefields, double* boundary_times, double* source_arr_in)
{
  printf("Entering elastic solver... \n");
  int ierr,ssind,ssi,nthr;
  int n_snapshot_occasions = 0;
  int n_recording_occasions = 0;
  int i_snap_var_out=0, i_trace_var_out=0; //Added by Bram. For indexing snapshot memory array.
  int i_snapshot_occasion=0, i_recording_occasion=0;
  double time,tmpd,force;
  double cputime0, cputimetic, cputimetoc;
  char logfile[256],sgram_file[256];
  FILE *fid, *sgramfid[NOUTVAR];

/**********************************************************/
  nthr = get_physical_cpu_count(); //If this fails on your platform, you can always hardcode it with the number of processors you have. Just replace with '4' for instance.
  omp_set_num_threads(nthr);
/**********************************************************/

  source_arr = source_arr_in; //Use global for convenience...

  cputime0   = omp_get_wtime();  
  cputimetic = omp_get_wtime();

  sprintf(logfile,"fd2dlog.asc");

  if( !(fid=fopen(logfile,"w")) ) {
      printf("log file not opened"); 
      return(1);
  }
  
  fprintf(fid,"\n ** Number of thread: %i. \n\n",nthr);
  printf("\n ** Number of thread: %i. \n\n",nthr);

  //ierr = readin_fd2d(fid);
  ierr = readin_from_info_struct(istruct);
  if(ierr) {
     fprintf(fid,"Error reading parameters, ierr = %i, see fd2dlog.asc \n",ierr);
     printf("Error reading parameters, ierr = %i, see fd2dlog.asc \n",ierr);
     cleanup_early(fid,ierr);
     fclose(fid);
     return(1);
  }
  fflush(fid);

 //=================================================================//
 // setup elastic constants cij 
 // the elastic constants rescaled as (c[i][j][k] = c[i][j][k] * dt / dx), except for the density 
  fprintf(fid,"starting setup stiffness matrix \n");
  printf("starting setup stiffness matrix \n");

  if(iGRID!=3) {
     ipro = i2matrix(mm,kk); // flag: 0 = isotropic; 1 = anisotropic;
  }
  c = d3matrix(mm,kk,4);  // nparmax = max # of elastic constants (9 cijs and dens) 

  ierr = setup_c(fid, vp, rho, vs);
  if(ierr) {
      fprintf(fid,"Error in setup C \n");
      printf("Error in setup C \n");
      cleanup_early(fid,ierr); 
      fclose(fid); 
      return(1);
  }   
  fprintf(fid,"stiffness matrix setup complete.\n\n");
  printf("stiffness matrix setup complete.\n\n");
  fflush(fid);

 //=================================================================//
 // allocate memory for stresses and velocities and PML damping values d in x,z direction 
  fprintf(fid,"starting allocate memory and initialize variables \n");
  printf("starting allocate memory and initialize variables \n");
  setup(fid);   
  fprintf(fid,"setup and allocate memory complete \n\n");
  printf("setup and allocate memory complete \n\n");

 //=================================================================//
 // calculate the PML damping values for grids within absorbing boundary 
   fprintf(fid,"Calculating damping values for absorbing boundaries.\n");
   setup_damping(fid); /* setup damping values */
   fprintf(fid,"setup damping value complete. \n\n");
   fflush(fid);

 //=================================================================//
 //  open and setup file to save history points 

  if (nsrc > 1) { //Edit Bram
	  printf("file naming below is designed for one shot per simulation");
	  exit(0);
  }

  if(traces_mem==0 && hist.l){ //if traces stored to disk
	  for ( k=0;k<NOUTVAR;k++ ) sgramfid[k] = 0;
	  for ( k=0;k<NOUTVAR;k++ ) {
		  sgramfid[k] = 0;
		  if ( histvar[k] ) {
			  switch (k) {
				case 0:  sprintf(sgram_file, "%ssx_%.2f_m_sz_%.2f_m_%s", traces_output_dir, srcxz[0][0], srcxz[1][0], "vx.bin");  break;
				case 1:  sprintf(sgram_file, "%ssx_%.2f_m_sz_%.2f_m_%s", traces_output_dir, srcxz[0][0], srcxz[1][0], "vz.bin");  break;
				case 2:  sprintf(sgram_file, "%ssx_%.2f_m_sz_%.2f_m_%s", traces_output_dir, srcxz[0][0], srcxz[1][0], "pr.bin");  break;
				case 3:  sprintf(sgram_file, "%ssx_%.2f_m_sz_%.2f_m_%s", traces_output_dir, srcxz[0][0], srcxz[1][0], "div.bin");  break;
				case 4:  sprintf(sgram_file, "%ssx_%.2f_m_sz_%.2f_m_%s", traces_output_dir, srcxz[0][0], srcxz[1][0], "curl.bin");  break;
				case 5:  sprintf(sgram_file, "%ssx_%.2f_m_sz_%.2f_m_%s", traces_output_dir, srcxz[0][0], srcxz[1][0], "txx.bin");  break;
				case 6:  sprintf(sgram_file, "%ssx_%.2f_m_sz_%.2f_m_%s", traces_output_dir, srcxz[0][0], srcxz[1][0], "tzz.bin");  break;
				case 7:  sprintf(sgram_file, "%ssx_%.2f_m_sz_%.2f_m_%s", traces_output_dir, srcxz[0][0], srcxz[1][0], "txz.bin");  break;
				case 8:  sprintf(sgram_file, "%ssx_%.2f_m_sz_%.2f_m_%s", traces_output_dir, srcxz[0][0], srcxz[1][0], "vpx.bin");  break;
				case 9:  sprintf(sgram_file, "%ssx_%.2f_m_sz_%.2f_m_%s", traces_output_dir, srcxz[0][0], srcxz[1][0], "vpz.bin");  break;
				case 10: sprintf(sgram_file, "%ssx_%.2f_m_sz_%.2f_m_%s", traces_output_dir, srcxz[0][0], srcxz[1][0], "vsx.bin");  break;
				case 11: sprintf(sgram_file, "%ssx_%.2f_m_sz_%.2f_m_%s", traces_output_dir, srcxz[0][0], srcxz[1][0], "vsz.bin");  break;
			  }
			  printf("Opening %s: \n", sgram_file);
			  if( !(sgramfid[k]=fopen(sgram_file,"wb")) ) {
				fprintf(fid,"seismogram file not opened");
				cleanup(fid);
				fclose(fid);
				for ( i=0;i<k;i++ ) {if ( sgramfid[i]!=0 ) fclose(sgramfid[i]);}
				return(1);
			  }
			  hist_setup(rcvxz,sgramfid[k]); /* write # of receiver and receiver coordinates */
		  }
	  }
  }
  fflush(fid);

  fprintf(fid,"\n\n");
  fprintf(fid,"starting time loop, number of steps = %i\n",itimestep);
  fprintf(fid,"-----------------------------------------\n\n");
  if(screen.l) {
     printf("starting time loop, number of steps = %i\n",itimestep);
     printf("-------------------------------------------\n\n");
  }

  /* on test runs, to quit here and not do time loop */
  if ( itimestep==0 ) {
	cleanup(fid);
	if(screen.l) printf("quitting before time loop\n");
	fclose(fid);
	if(traces_mem==0 && hist.l){//if we opened trace files
		for ( k=0;k<NOUTVAR;k++ ) {if ( sgramfid[k]!=0 ) fclose(sgramfid[k]);}
	}
	return(1);
   }

/*********************************************************************
 *  do the time loop                                                 */

  if (  screen.l ) {
	cputimetoc = omp_get_wtime();
        tmpd = cputimetoc - cputimetic;
	fprintf(fid,"CPU time for setup: %f (sec), \n",tmpd);
	printf("CPU time for setup: %f (sec), \n",tmpd);
	cputimetic = omp_get_wtime();
   }

  fflush(fid);

  ssind = 0;

  //First determine how many recording occasions, also populate time array
  for(jj=0;jj<itimestep;jj++) {
	  time = dt*jj;
	  if (hist.l && traces_mem && (jj>=hist.ii) && (jj<=hist.ie) && (jj % hist.di)==0){ //only if we pass to pysit
		  shotgather_times[n_recording_occasions] = time;
		  n_recording_occasions = n_recording_occasions + 1;
	  }

	  if(snap.l && snaps_mem && (jj >= snap.ii) && (jj <= snap.ie) && ((jj-snap.ii) % snap.di) == 0) { //only if we pass to pysit
		  wavefield_times[n_snapshot_occasions] = time;
		  n_snapshot_occasions = n_snapshot_occasions + 1;
	  }
  }

  printf("Nr of trace steps %i: Nr of snaps %i \n", n_recording_occasions, n_snapshot_occasions);

  for(jj=0;jj<itimestep;jj++) {
      time = dt*jj;
      switch(iGRID) {
           case 0:  // 4th order SSG
        	   update_T_SSG(jj,fid, boundary_wavefields);
        	   update_V_SSG(jj,fid, boundary_wavefields);
        	   if (bparams.rec_boundary){//only implemented for fourth order SSG
        		   boundary_times[jj] = time;
        		   rec_wavefields(boundary_wavefields);
        	   }
        	   break;
           case 1:  // 2nd order RSG
        	   update_T_RSG2(jj,fid);
        	   update_V_RSG2(jj,fid);
        	   break;
           case 2:  // 4th order RSG
        	   update_T_RSG4(jj,fid);
        	   update_V_RSG4(jj,fid);
        	   break;
            case 3: // Acoustic (4th order SSG)
            	update_T_SSG_Acoustic(jj,fid);
            	update_V_SSG_Acoustic(jj,fid);
            	break;
      }
       if(hist.l && (jj>=hist.ii) && (jj<=hist.ie) && (jj % hist.di)==0) {
    	   if(traces_mem==0){ //if we store traces on disk
			  for(ssi=0;ssi<NOUTVAR;ssi++) {
				  if(histvar[ssi]) {
					  for(m=0;m<nhist;m++)  rcvval[m] = (float)output_var(ssi,rcvind[0][m],rcvind[1][m]);
					  	  fwrite(&rcvval[0], sizeof(float), nhist, sgramfid[ssi]);
				  }
			  }
    	   }
    	   else{ //save trace to memory
    		   for(ssi=0;ssi<NOUTVAR;ssi++) {
    			   if(histvar[ssi]) {
    				   int offset_var =  i_trace_var_out * (n_recording_occasions * nhist);
    				   for (k=0;k<nhist;k++){//for each receiver
    					   int offset_receiver = k*n_recording_occasions;
    					   int offset = offset_var + offset_receiver;
    					   shotgather_mem[offset + i_recording_occasion] = output_var(ssi,rcvind[0][k],rcvind[1][k]);
    				   }
					   i_trace_var_out = i_trace_var_out + 1;

    			   }
    		   }
    		   i_recording_occasion = i_recording_occasion + 1;
    	   }
        }
       // output snapshots 
       if(snap.l && (jj >= snap.ii) && (jj <= snap.ie) && ((jj-snap.ii) % snap.di) == 0) {
          ssind = ssind+1;
          fprintf(fid,"step %i of %i: snapshot out\n",jj+1,itimestep);
          if(screen.l) printf("step %i of %i: snapshot out\n",jj+1,itimestep);


          for(ssi=0;ssi<NOUTVAR;ssi++) {            
              if(ss.var[ssi]) { 
                 if(snaps_mem==0){ //Save to disk
                	 ssval = f1matrix(ss.M*ss.K);
                     for(k=0;k<ss.K;k++)
                         for(m=0;m<ss.M;m++)
                             ssval[k + m*ss.K] = (float)output_var(ssi,ss.m0 + m*ss.dm, ss.k0 + k*ss.dk);
					 SSfid = setupss(ssi,ssind,fid,time);

					 if(SSfid==0) {
						fprintf(fid,"error opening snapshot file \n");
						if(screen.l) printf("error opening snapshot file \n");
						if(traces_mem==0 && hist.l){ //if we saved traces to disk as well, close these files
							for ( k=0;k<NOUTVAR;k++ ) {if ( sgramfid[k]!=0 ) fclose(sgramfid[k]);}
						}
						cleanup(fid); fclose(fid);
						return(1);
					 }
					 fwrite(&ssval[0],sizeof(float),ss.M*ss.K,SSfid);
					 fclose(SSfid);
					 free_1f(ssval);
                 }
                 else{ //save to mem, assume output array has been allocated
                	 int offset_var =  i_snap_var_out * (n_snapshot_occasions * ss.K * ss.M);
                	 int offset_time = i_snapshot_occasion * (ss.K * ss.M); //every timestep we save the offset increases by the number of nodes we save (ss.K * ss.M)
                	 int offset = offset_var + offset_time;
                     for(k=0;k<ss.K;k++)
                         for(m=0;m<ss.M;m++){
                        	 wavefield_mem[offset + k*ss.M + m] = output_var(ssi,ss.m0 + m*ss.dm, ss.k0 + k*ss.dk);
                         }
                     i_snap_var_out = i_snap_var_out + 1;
                 }
              } // end of if ss.var  
          } // of ssi loop 
          i_snapshot_occasion  = i_snapshot_occasion  + 1;
       } // end of if snap.l ...       

       if((jj>=screen.ii) && (jj<=screen.ie) && (jj % screen.di)==0) {
           cputimetoc = omp_get_wtime();
           tmpd = cputimetoc - cputimetic;
           if(tmpd<100.) {
              fprintf(fid,"step %i (of %i): CPU time since last: %.2f (sec), \n",jj,itimestep,tmpd);
              if(screen.l) printf("step %i (of %i): CPU time since last: %.2f (sec), \n",jj,itimestep,tmpd);
           }
           else {
              fprintf(fid,"step %i (of %i): CPU time since last: %.2f (min), \n",jj,itimestep,tmpd/60.);
              if(screen.l) printf("step %i (of %i): CPU time since last: %.2f (min), \n",jj,itimestep,tmpd/60.);
           }
           cputimetic = omp_get_wtime();
	   fflush(fid);
       }
       i_snap_var_out = 0; //reset for next loop
       i_trace_var_out = 0;
  }  // end of jj time loop

  /*****************************************************************/

    fprintf(fid,"time loop finished.\n\n");
    if(screen.l) printf("time loop finished.\n\n");
    cputimetoc = omp_get_wtime();
    tmpd = cputimetoc - cputime0;
    if(tmpd<100.) {
       fprintf(fid,"total time = %.2f (sec). \n",tmpd);
       if(screen.l) printf("total time = %.2f (sec). \n",tmpd);
    }
    else if(tmpd<3600.) {
       fprintf(fid,"total time = %.2f (min). \n",tmpd/60.);
       if(screen.l) printf("total time = %.2f (min). \n",tmpd/60.);
    }
    else {
       fprintf(fid,"total time = %.2f (hour). \n",tmpd/3600.);
       if(screen.l) printf("total time = %.2f (hour). \n",tmpd/3600.);
    }
    cleanup(fid);
    if(traces_mem==0 && hist.l){ //if we saved traces to disk, close files
		for ( k=0;k<NOUTVAR;k++ ) {if ( sgramfid[k]!=0 ) fclose(sgramfid[k]);}
		fprintf(fid,"---- done ----\n\n");
		if(screen.l) printf("---- done ----\n\n");
    }

    fprintf(fid,"closing log file.\n\n");
    if(screen.l) printf("closing log file.\n\n");
    fclose(fid);

    printf("Leaving elastic solver...\n");
    return(0);
}



