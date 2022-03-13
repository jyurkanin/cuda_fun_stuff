#include <SDL2/SDL.h>
#include <stdio.h>
//#include <math.h>

//#include "cool.h"


const int SCREEN_WIDTH = 640;
const int SCREEN_HEIGHT = 640;

//abs((int)x[i]-960)*abs((int)y[i]-540)/param;

__global__
void compute_pixel(int n, int param, float *x, float *y, unsigned int *pixel){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for(int i = idx; i < n; i += stride){
        float z1 = x[i];
        float z2 = y[i];
        float dz1;
        float dz2;
        for(int j = 0; j < 1000; j++){
            dz1 = z1 + z2;
            dz2 = 1.0f/(param*z1*z2) - (z2*z1)/((float)param);
            z1 += dz1*.01;
            z2 += dz2*.01;
        }
        
        pixel[i] = (int)0xFF*(z1+z2);
    }
}



//https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html
int main( int argc, char* args[]){
    //The window we'll be rendering to
    SDL_Window* window = NULL;
  
    if(SDL_Init( SDL_INIT_VIDEO ) < 0){
        printf( "SDL could not initialize! SDL_Error: %s\n", SDL_GetError() );
        SDL_DestroyWindow( window );
        SDL_Quit();
        return 0;
    }
    
    SDL_DisplayMode dm;
    if(SDL_GetDesktopDisplayMode(0, &dm) != 0){
        SDL_Log("SDL_GetDesktopDisplayMode failed: %s", SDL_GetError());
        return 1;
    }
    
    int width = dm.w;
    int height = dm.h;
    
    window = SDL_CreateWindow( "Shit Poop", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, width, height, SDL_WINDOW_SHOWN );
    if(window == NULL) {
        printf("Window could not be created! SDL_Error: %s\n", SDL_GetError());
        SDL_DestroyWindow( window );
        SDL_Quit();
        return 0;
    }
  
    SDL_SetWindowFullscreen(window, SDL_WINDOW_FULLSCREEN);
    
    
    SDL_Surface *window_surface = SDL_GetWindowSurface(window);
    unsigned int *pixels = (unsigned int *) window_surface->pixels;
    
    
    unsigned num_pixels = width*height;
    printf("Height %d     Width %d\n", height, width);
    printf("Pixel format: %s\n", SDL_GetPixelFormatName(window_surface->format->format));
    
    int count;
    int device;
    cudaGetDeviceCount(&count);
    cudaGetDevice(&device);
    printf("number of devices %d\n", count);
    printf("My device is #%d\n", device);
    
    unsigned int *cuda_pixel_buf = 0; //stores result.
    float *cuda_x_buf;
    float *cuda_y_buf;
    
    cudaMallocManaged(&cuda_pixel_buf, num_pixels*sizeof(unsigned int));
    cudaMallocManaged(&cuda_x_buf, num_pixels*sizeof(float));
    cudaMallocManaged(&cuda_y_buf, num_pixels*sizeof(float));
    
    //printf("cuda error %d\n", error);
    
    unsigned int cnt = 0;
    for(int y = 0; y < height; y++){
        for(int x = 0; x < width; x++){
            cuda_x_buf[cnt] = (x/(float)width) - (1/2.0f);
            cuda_y_buf[cnt] = (y/(float)height) - (1/2.0f);
            cnt++;
        }
    }
    
    int param = 0;
    while(1){
        SDL_Event event;
        while(SDL_PollEvent(&event)) {
            if(event.type == SDL_QUIT || (event.type == SDL_KEYDOWN && event.key.keysym.sym == SDLK_ESCAPE)){
                SDL_DestroyWindow( window );
                SDL_Quit();
                cudaFree(cuda_pixel_buf);
                cudaFree(cuda_x_buf);
                cudaFree(cuda_y_buf);
                return 0;
            }
            if(event.type == SDL_WINDOWEVENT) {
                if(event.window.event == SDL_WINDOWEVENT_SIZE_CHANGED) {
                    window_surface = SDL_GetWindowSurface(window);
                    pixels = (unsigned int *) window_surface->pixels;
                    width = window_surface->w;
                    height = window_surface->h;
                    printf("Size changed: %d, %d\n", width, height);
                }
            }
        }
        
        int blockSize = 256;
        int numBlocks = (int) ceilf(num_pixels / blockSize);
        printf("param %d\n", param);
        compute_pixel<<<numBlocks, blockSize>>>(num_pixels, param, cuda_x_buf, cuda_y_buf, cuda_pixel_buf);
        cudaDeviceSynchronize();
        
        for(int ii = 0; ii < num_pixels; ii++){
            pixels[ii] = cuda_pixel_buf[ii];
        }
        param++;
        SDL_UpdateWindowSurface(window);
    }
    
    SDL_DestroyWindow(window);
    SDL_Quit();
    cudaFree(cuda_pixel_buf);
    cudaFree(cuda_x_buf);
    cudaFree(cuda_y_buf);
    return 0;
}
