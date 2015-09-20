#include <curand_kernel.h>

extern "C" __global__ void addVector(int* a, int aLen0, int* b, int bLen0, int* c, int cLen0)
{
	int x = blockIdx.x;
	c[x] = a[x] + b[x];
}

extern "C" __global__ void subVector(int* a, int aLen0, int* b, int bLen0, int* c, int cLen0)
{
	int x = blockIdx.x;
	c[x] = a[x] - b[x];
}

extern "C" __global__ void mulVector(int* a, int aLen0, int* b, int bLen0, int n)
{
	int x = blockIdx.x;
	b[x] = a[x] * n;
}

extern "C" __global__ void divVector(int* a, int aLen0, int* b, int bLen0, int n)
{
	int x = blockIdx.x;
	b[x] = a[x] / n;
}
