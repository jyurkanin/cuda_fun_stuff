#include <SDL2/SDL.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

//#include "cool.h"

#define GS (0x100)
#define KR (1)
#define KS (1 + (2*KR))


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
void computeODE(int n, float *X, float *Xd, float *K){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for(int i = idx; i < n; i += stride){
      int x = i & 0xFF;
      int y = (i & 0xFF00) >> 8;
      
      Xd[i] = 0;

      //x + kx - KR
      int off_x = x - KR;
      int off_y = y - KR;

      for(int ky = 0; ky < KS; ky++){
        for(int kx = 0; kx < KS; kx++){
          int tempx = (GS + (kx + off_x)) & 0xFF; //This is equivalent to a modulus.
          int tempy = (GS + (ky + off_y)) & 0xFF;
          
          int k_idx = (ky*KS) + kx;
          int x_idx = (tempy*GS) + tempx;
          Xd[i] += K[k_idx]*X[x_idx];
        }
      }
    }
}

__global__
void get_pixels(int n, float *X, unsigned *pixels){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  
  for(int i = idx; i < n; i += stride){
    //pixels[i] = (unsigned) (0x800000 + (0x100*X[i]));
    pixels[i] = 0x10000*X[i] + 0xFF0000;
  }
}


__global__
void updateState(int n, float *X, float *Xd){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  float ts = .01;
  for(int i = idx; i < n; i += stride){
    X[i] += ts*Xd[i];
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
    
    int width = GS; //dm.w;
    int height = GS; //dm.h;
    
    window = SDL_CreateWindow( "Shit Poop", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, width, height, SDL_WINDOW_SHOWN );
    if(window == NULL) {
        printf("Window could not be created! SDL_Error: %s\n", SDL_GetError());
        SDL_DestroyWindow( window );
        SDL_Quit();
        return 0;
    }
  
    //SDL_SetWindowFullscreen(window, SDL_WINDOW_FULLSCREEN);
    
    
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
    float *cuda_X;
    float *cuda_Xd;
    float *cuda_K;
    
    cudaMallocManaged(&cuda_pixel_buf, num_pixels*sizeof(unsigned int));
    cudaMallocManaged(&cuda_X, num_pixels*sizeof(float));
    cudaMallocManaged(&cuda_Xd, num_pixels*sizeof(float));
    cudaMallocManaged(&cuda_K, KS*KS*sizeof(float));

    float k1 = 1;
    float k2 = .5;
    float temp_value[KS*KS] = {
                               0, k1, k2,
                               -k1, 0, k1,
                               -k2, -k1, 0
    };
    for(int i = 0; i < KS*KS; i++){
      cuda_K[i] = temp_value[i];
    }
    
    
    unsigned int cnt = 0;
    for(int y = 0; y < height; y++){
        for(int x = 0; x < width; x++){
          cuda_X[cnt] = 0; //(rand() / (float)RAND_MAX);
          cnt++;
        }
    }

    int rradius = 8;
    for(int i = -rradius; i <= rradius; i++){
      for(int j = -rradius; j <= rradius; j++){
        cuda_X[((127+i)*GS) + 127+j] = 1;
        //cuda_X[((127+i)*GS) + 127+j] += (param*rand() / (float)RAND_MAX);
      }
    }
    
    //cuda_X[(GS*GS/2) + (GS/2)] = 1;
    
    float param = 4;
    int paused = 1;
    while(1){
        SDL_Event event;
        while(SDL_PollEvent(&event)) {
          if(event.type == SDL_KEYDOWN ){
            if(event.key.keysym.sym == SDLK_ESCAPE){
              SDL_DestroyWindow( window );
              SDL_Quit();
              
              cudaFree(cuda_pixel_buf);
              cudaFree(cuda_X);
              cudaFree(cuda_Xd);
              cudaFree(cuda_K);
              return 0;
            }
            else if(event.key.keysym.sym == SDLK_RETURN){
              paused = !paused;
            }
          }
          if(event.type == SDL_QUIT){
            SDL_DestroyWindow( window );
            SDL_Quit();
            
            cudaFree(cuda_pixel_buf);
            cudaFree(cuda_X);
            cudaFree(cuda_Xd);
            cudaFree(cuda_K);
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
        
        if(paused){
          usleep(10000);
          continue;
        }
        
        int blockSize = 256;
        int numBlocks = (int) ceilf(num_pixels / blockSize);
        //printf("param %f\n", param);
        computeODE<<<numBlocks, blockSize>>>(num_pixels, cuda_X, cuda_Xd, cuda_K);
        updateState<<<numBlocks, blockSize>>>(num_pixels, cuda_X, cuda_Xd);
        get_pixels<<<numBlocks, blockSize>>>(num_pixels, cuda_X, cuda_pixel_buf);
        cudaDeviceSynchronize();
        
        printf("0,0: %f\n", cuda_X[0]);
        
        //int rradius = 8;
        for(int i = -rradius; i <= rradius; i++){
          for(int j = -rradius; j <= rradius; j++){ 
            //cuda_X[((127+i)*GS) + 127+j] += (param*rand() / (float)RAND_MAX);
          }
        }

        
        for(int ii = 0; ii < num_pixels; ii++){
          pixels[ii] = cuda_pixel_buf[ii];
        }
        
        //param = .01f;
        //
        SDL_UpdateWindowSurface(window);
    }
    
    SDL_DestroyWindow(window);
    SDL_Quit();
    
    cudaFree(cuda_pixel_buf);
    cudaFree(cuda_X);
    cudaFree(cuda_Xd);
    cudaFree(cuda_K);
    return 0;
}
