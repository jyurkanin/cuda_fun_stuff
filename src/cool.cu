#include <SDL2/SDL.h>
#include <stdio.h>
//#include <math.h>

//#include "cool.h"


const int SCREEN_WIDTH = 640;
const int SCREEN_HEIGHT = 640;

__device__
void HSVtoRGB(float& fR, float& fG, float& fB, float& fH, float& fS, float& fV) {
  float fC = fV * fS; // Chroma
  float fHPrime = fmod(fH / 60.0f, 6.0f);
  float fX = fC * (1 - fabs(fmod(fHPrime, 2.0f) - 1));
  float fM = fV - fC;
  
  if(0 <= fHPrime && fHPrime < 1) {
    fR = fC;
    fG = fX;
    fB = 0;
  } else if(1 <= fHPrime && fHPrime < 2) {
    fR = fX;
    fG = fC;
    fB = 0;
  } else if(2 <= fHPrime && fHPrime < 3) {
    fR = 0;
    fG = fC;
    fB = fX;
  } else if(3 <= fHPrime && fHPrime < 4) {
    fR = 0;
    fG = fX;
    fB = fC;
  } else if(4 <= fHPrime && fHPrime < 5) {
    fR = fX;
    fG = 0;
    fB = fC;
  } else if(5 <= fHPrime && fHPrime < 6) {
    fR = fC;
    fG = 0;
    fB = fX;
  } else {
    fR = 0;
    fG = 0;
    fB = 0;
  }
  
  fR += fM;
  fG += fM;
  fB += fM;
}

__global__
void compute_pixel(int n, float param, float *x, float *y, unsigned int *pixel, unsigned int *prev_pixel){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    float red;
    float green;
    float blue;
    
    unsigned red_u;
    unsigned green_u;
    unsigned blue_u;
    
    float hue;
    float sat = 1;
    float val = 1;
    
    for(int i = idx; i < n; i += stride){
        float r1 = x[i];
        float r2 = y[i];
        float nr1;
        int j;
        for(j = 0; j < 1000 && ((r1*r1)+(r2*r2)) < 36; j++){
            r1 =  r1 + param/r2; //looks like a sunset around param <= -7
            r2 = -r2 + 1/r1 + 1/r2;
            r1 = nr1;
        }

        //hue = 350.0f*j/1000;
        red_u = (unsigned) 85*log10f(1 + (float)j);
        green_u = 0;
        blue_u = 0;
        pixel[i] = (red_u << 16) + (green_u << 8) + blue_u;
        
        
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
    unsigned int *cuda_pixel_buf_prev = 0;
    float *cuda_x_buf;
    float *cuda_y_buf;
    
    cudaMallocManaged(&cuda_pixel_buf, num_pixels*sizeof(unsigned int));
    cudaMallocManaged(&cuda_pixel_buf_prev, num_pixels*sizeof(unsigned int));
    cudaMallocManaged(&cuda_x_buf, num_pixels*sizeof(float));
    cudaMallocManaged(&cuda_y_buf, num_pixels*sizeof(float));
    
    //printf("cuda error %d\n", error);
    
    unsigned int cnt = 0;
    for(int y = 0; y < height; y++){
        for(int x = 0; x < width; x++){
            cuda_x_buf[cnt] = (10*x/(float)width) - 5.0f;
            cuda_y_buf[cnt] = (10*y/(float)height) - 5.0f;
            cuda_pixel_buf_prev[cnt] = 0;
            cnt++;
        }
    }
    
    float param = 1;
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
        printf("param %f\n", param);
        compute_pixel<<<numBlocks, blockSize>>>(num_pixels, param, cuda_x_buf, cuda_y_buf, cuda_pixel_buf, cuda_pixel_buf_prev);
        cudaDeviceSynchronize();

        pixels[0] = cuda_pixel_buf[0];
        cuda_pixel_buf_prev[0] = cuda_pixel_buf[num_pixels-1];
        for(int ii = 1; ii < num_pixels; ii++){
            pixels[ii] = cuda_pixel_buf[ii];
            cuda_pixel_buf_prev[ii] = cuda_pixel_buf[ii-1];
            
        }
        param -= .01f;
        SDL_UpdateWindowSurface(window);
    }
    
    SDL_DestroyWindow(window);
    SDL_Quit();
    cudaFree(cuda_pixel_buf);
    cudaFree(cuda_x_buf);
    cudaFree(cuda_y_buf);
    return 0;
}
