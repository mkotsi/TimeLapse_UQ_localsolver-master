//http://stackoverflow.com/questions/2901694/programatically-detect-number-of-physical-processors-cores-or-if-hyper-threading
#ifdef _WIN32
	#include <windows.h>
#elif MACOS
	#include <sys/param.h>
	#include <sys/sysctl.h>
#else
	#include <unistd.h>
#endif
#include <stdio.h>

int get_physical_cpu_count(){
	int registers[4];
	unsigned logicalcpucount;
	unsigned physicalcpucount;
	#ifdef _WIN32
	SYSTEM_INFO systeminfo;
	GetSystemInfo( &systeminfo );

	logicalcpucount = systeminfo.dwNumberOfProcessors;

	#else
	logicalcpucount = sysconf( _SC_NPROCESSORS_ONLN );
	#endif

	__asm__ __volatile__ ("cpuid " :
						  "=a" (registers[0]),
						  "=b" (registers[1]),
						  "=c" (registers[2]),
						  "=d" (registers[3])
						  : "a" (1), "c" (0));

	unsigned CPUFeatureSet = registers[3];
	int hyperthreading = CPUFeatureSet & (1 << 28);

	if (hyperthreading){
		physicalcpucount = logicalcpucount / 2;
	} else {
		physicalcpucount = logicalcpucount;
	}

	printf("LOGICAL CPU COUNT: %i\n", logicalcpucount);
	printf("PHYSICAL CPU COUNT: %i\n", physicalcpucount);
	return(physicalcpucount);
}
