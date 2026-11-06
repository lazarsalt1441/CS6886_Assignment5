/* STREAM-like triad, 384 MiB footprint; 12 useful bytes/element.
 * Sustained application bandwidth, not theoretical DRAM channel peak.
 * Cached stores may incur additional uncounted write allocation traffic. */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
int main(void) {
    const size_t n=(size_t)32*1024*1024;
    float *a=aligned_alloc(64,n*4),*b=aligned_alloc(64,n*4),*c=aligned_alloc(64,n*4);
    if(!a||!b||!c) return 1;
    #pragma omp parallel for schedule(static)
    for(size_t i=0;i<n;++i) {a[i]=0;b[i]=1+(i%13)*0.01f;c[i]=2;}
    double times[7];
    for(int rep=0;rep<8;++rep) {
        double t=omp_get_wtime();
        #pragma omp parallel for schedule(static)
        for(size_t i=0;i<n;++i) a[i]=b[i]+3.0f*c[i];
        t=omp_get_wtime()-t;
        if(rep) times[rep-1]=t;
        b[rep]+=a[rep]*1e-8f;
    }
    for(int i=0;i<7;++i) for(int j=i+1;j<7;++j)
        if(times[j]<times[i]) {double t=times[i];times[i]=times[j];times[j]=t;}
    double checksum=0;
    #pragma omp parallel for reduction(+:checksum)
    for(size_t i=0;i<n;++i) checksum+=a[i];
    printf("threads=%d median_s=%.9f bandwidth_gbs=%.6f checksum=%.6f\n",omp_get_max_threads(),times[3],12.0*n/times[3]*1e-9,checksum);
    free(a);free(b);free(c);return 0;
}
