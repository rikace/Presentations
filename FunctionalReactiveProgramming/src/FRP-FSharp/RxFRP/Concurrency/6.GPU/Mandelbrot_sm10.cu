#include <stdio.h>

template<class T>
__device__ inline int CalcMandelbrot(const T xPos, const T yPos, const int crunch)
{
    T y = yPos;
    T x = xPos;
    T yy = y * y;
    T xx = x * x;
    int i = crunch;

    while (--i && (xx + yy < T(4.0))) {
        y = x * y * T(2.0) + yPos;
        x = xx - yy + xPos;
        yy = y * y;
        xx = x * x;
    }
    return i; 
} 


// The Mandelbrot CUDA GPU thread function
extern "C" __global__ void Mandelbrot0_sm10(int *dst, const int imageW, const int imageH, const int crunch, 
											const float xOff, const float yOff, const float scale)
{
    const int ix = blockDim.x * blockIdx.x + threadIdx.x;
    const int iy = blockDim.y * blockIdx.y + threadIdx.y;

    if ((ix < imageW) && (iy < imageH)) {
		// Calculate the location
		const float xPos = xOff + (float)ix * scale;
		const float yPos = yOff - (float)iy * scale;
		
        // Calculate the Mandelbrot index for the current location
        int m = CalcMandelbrot<float>(xPos, yPos, crunch);
        m = m > 0 ? crunch - m : crunch;
			
        // Output the pixel
 		int pixel = imageW * iy + ix;
		dst[pixel] = m;
    }
} 

