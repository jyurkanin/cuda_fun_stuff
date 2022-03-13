#include <SDL2/SDL.h>
#include <stdio.h>

//#include "cool.h"


const int SCREEN_WIDTH = 640;
const int SCREEN_HEIGHT = 640;


__global__
void compute_pixel(int n, unsigned int *x, unsigned int *y, unsigned int *pixel){
  for(int i = 0; i < n; i++)
    pixel[i] = x[i]^y[i];
}



int main( int argc, char* args[]){
  //The window we'll be rendering to
  SDL_Window* window = NULL;
  
  if(SDL_Init( SDL_INIT_VIDEO ) < 0){
    printf( "SDL could not initialize! SDL_Error: %s\n", SDL_GetError() );
    SDL_DestroyWindow( window );
    SDL_Quit();
    return 0;
  }
  
  
  window = SDL_CreateWindow( "Shit Poop", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, SCREEN_WIDTH, SCREEN_HEIGHT, SDL_WINDOW_SHOWN );
  if(window == NULL) {
    printf("Window could not be created! SDL_Error: %s\n", SDL_GetError());
    SDL_DestroyWindow( window );
    SDL_Quit();
  }
  
  
  SDL_Surface * window_surface = SDL_GetWindowSurface(window);
  unsigned int * pixels = (unsigned int *) window_surface->pixels;
  int width = window_surface->w;
  int height = window_surface->h;
  unsigned num_pixels = width*height;
  printf("Pixel format: %s\n", SDL_GetPixelFormatName(window_surface->format->format));
  
  int count;
  int device;
  cudaGetDeviceCount(&count);
  cudaGetDevice(&device);
  printf("number of devices %d\n", count);
  printf("My device is #%d\n", device);
  
  unsigned int *cuda_pixel_buf = 0; //stores result.
  unsigned int *cuda_x_buf;
  unsigned int *cuda_y_buf;

  cudaMallocManaged(&cuda_pixel_buf, num_pixels*sizeof(unsigned int));
  cudaError_t error = cudaMallocManaged(&cuda_x_buf, num_pixels*sizeof(unsigned int));
  cudaMallocManaged(&cuda_y_buf, num_pixels*sizeof(unsigned int));
  
  printf("cuda error %d\n", error);
  
  while(1){
    SDL_Event event;
    while (SDL_PollEvent(&event)) {
      if (event.type == SDL_QUIT){
        SDL_DestroyWindow( window );
        SDL_Quit();
        cudaFree(cuda_pixel_buf);
        cudaFree(cuda_x_buf);
        cudaFree(cuda_y_buf);
      }
      if (event.type == SDL_WINDOWEVENT) {
        if (event.window.event == SDL_WINDOWEVENT_SIZE_CHANGED) {
          window_surface = SDL_GetWindowSurface(window);
          pixels = (unsigned int *) window_surface->pixels;
          width = window_surface->w;
          height = window_surface->h;
          printf("Size changed: %d, %d\n", width, height);
        }
      }
    }
    
    unsigned int cnt = 0;
    for(int y = 0; y < height; y++){
      for(int x = 0; x < width; x++){
        cuda_x_buf[cnt] = x;
        cuda_y_buf[cnt] = y;
        cnt++;
      }
    }
    
    compute_pixel<<<1, 1>>>(num_pixels, cuda_x_buf, cuda_y_buf, cuda_pixel_buf);
    cudaDeviceSynchronize();
    
    cnt = 0;
    for(int y = 0; y < height; y++){
      for(int x = 0; x < width; x++){
        pixels[x + (y*width)] = cuda_pixel_buf[cnt];
        cnt++;
      }
    }
    
    
    SDL_UpdateWindowSurface(window);
  }
    
  return 0;
}
