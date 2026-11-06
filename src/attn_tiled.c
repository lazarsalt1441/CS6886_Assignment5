#define _POSIX_C_SOURCE 200809L
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif

/* Q3: tiled; driver below is unchanged from the starter. */
void attention(const float *Q,const float *K,const float *V,float *O,int N,int d)
{
    const float scale=1.0f/sqrtf((float)d);
    float *S=calloc((size_t)N*N,sizeof(float));

    float *T=malloc((size_t)d*N*sizeof(float));
    if (!S || !T) { fprintf(stderr,"allocation failed\n"); exit(1); }
    for(int j=0;j<N;++j) for(int k=0;k<d;++k) T[(size_t)k*N+j]=K[(size_t)j*d+k];

    /* 4x32 register tile, enclosed in 32x128 query/key cache tiles.
       Keeping the whole d reduction in local accumulators avoids repeated
       loads/stores of S. Scalar C permits compiler vectorisation. */
    for(int ii=0;ii<N;ii+=32) for(int jj=0;jj<N;jj+=128)
      for(int i=ii;i<N && i<ii+32;i+=4) for(int j=jj;j<N && j<jj+128;j+=32) {
        float a[4][32]={{0}};
        int nr=N-i<4?N-i:4, nc=N-j<32?N-j:32;
        for(int k=0;k<d;++k) {
            const float *t=T+(size_t)k*N+j;
            for(int r=0;r<nr;++r) {
                float q=Q[(size_t)(i+r)*d+k];
                for(int c=0;c<nc;++c) a[r][c]+=q*t[c];
            }
        }
        for(int r=0;r<nr;++r) for(int c=0;c<nc;++c)
            S[(size_t)(i+r)*N+j+c]=a[r][c]*scale;
      }

    for (int i=0; i<N; ++i) {
        float *row=S+(size_t)i*N, mx=row[0], sum=0;
        for (int j=1; j<N; ++j) if(row[j]>mx) mx=row[j];
        for (int j=0; j<N; ++j) { row[j]=expf(row[j]-mx); sum+=row[j]; }
        for (int j=0; j<N; ++j) row[j]/=sum;
    }

    memset(O,0,(size_t)N*d*sizeof(float));
    /* Four output rows reuse each contiguous V vector. */
    for(int i=0;i<N;i+=4) for(int kk=0;kk<d;kk+=32) {
        float a[4][32]={{0}};
        int nr=N-i<4?N-i:4,nc=d-kk<32?d-kk:32;
        for(int jj=0;jj<N;jj+=128) for(int j=jj;j<N&&j<jj+128;++j)
          for(int r=0;r<nr;++r) {
            float p=S[(size_t)(i+r)*N+j];
            for(int c=0;c<nc;++c) a[r][c]+=p*V[(size_t)j*d+kk+c];
          }
        for(int r=0;r<nr;++r) for(int c=0;c<nc;++c)
            O[(size_t)(i+r)*d+kk+c]=a[r][c];
    }
    free(T); free(S);
}

/* ------------------------------------------------------------------------- */
/* Driver: do not change below this line.                                     */
/* ------------------------------------------------------------------------- */
#define REPS 5

static double now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

static int cmp_double(const void *a, const void *b)
{
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

int main(int argc, char **argv)
{
    if (argc != 3) {
        fprintf(stderr, "usage: %s <input.bin> <output.bin>\n", argv[0]);
        return 1;
    }
    FILE *in = fopen(argv[1], "rb");
    if (!in) {
        perror(argv[1]);
        return 1;
    }
    int32_t hdr[2];
    if (fread(hdr, sizeof hdr, 1, in) != 1) {
        fprintf(stderr, "%s: bad header\n", argv[1]);
        return 1;
    }
    int N = hdr[0], d = hdr[1];
    size_t n = (size_t)N * d;
    float *Q = malloc(n * sizeof(float)), *K = malloc(n * sizeof(float));
    float *V = malloc(n * sizeof(float)), *O = malloc(n * sizeof(float));
    if (!Q || !K || !V || !O || fread(Q, sizeof(float), n, in) != n ||
        fread(K, sizeof(float), n, in) != n || fread(V, sizeof(float), n, in) != n) {
        fprintf(stderr, "%s: could not read Q, K, V\n", argv[1]);
        return 1;
    }
    fclose(in);

    attention(Q, K, V, O, N, d); /* warm-up, not timed */
    double t[REPS];
    for (int r = 0; r < REPS; r++) {
        double t0 = now();
        attention(Q, K, V, O, N, d);
        t[r] = now() - t0;
    }
    qsort(t, REPS, sizeof t[0], cmp_double);
    double median = t[REPS / 2];

    int threads = 1;
#ifdef _OPENMP
    threads = omp_get_max_threads();
#endif
    const char *name = strrchr(argv[0], '/') ? strrchr(argv[0], '/') + 1 : argv[0];
    printf("%-18s N=%-5d d=%-4d threads=%-3d time_ms=%10.3f  GFLOP/s=%8.2f\n", name, N, d,
           threads, median * 1e3, 4.0 * N * (double)N * d / median * 1e-9);

    FILE *out = fopen(argv[2], "wb");
    if (!out || fwrite(hdr, sizeof hdr, 1, out) != 1 || fwrite(O, sizeof(float), n, out) != n) {
        perror(argv[2]);
        return 1;
    }
    fclose(out);
    free(Q);
    free(K);
    free(V);
    free(O);
    return 0;
}
