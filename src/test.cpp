#include <SDL2/SDL.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>


#define GS (0x100)
#define KR (1)
#define KS (1 + (2*KR))


int main(){
  int num = GS*GS;
  float *X = new float[num];
  float *Xd = new float[num];
  float K[9] = {1,1,1, 1,1,1, 1,1,1};
  
  //for(int i = 0; i < (256*10)+10; i++){
  {
    int i = (256*10)+10;
    int x = i & 0xFF;
    int y = (i & 0xFF00) >> 8;
    
    printf("grid %d %d\n", x, y);
    
    Xd[i] = 0;
    
    //x + kx - KR
    int off_x = x - KR;
    int off_y = y - KR;
    
    for(int ky = 0; ky < KS; ky++){
      for(int kx = 0; kx < KS; kx++){
        int tempx = (GS + (kx + off_x)) & 0xFF; //This is equivalent to a modulus.
        int tempy = (GS + (ky + off_y)) & 0xFF;
        
        printf("temp: %d %d\n", tempx, tempy);
        
        int k_idx = (ky*KS) + kx;
        int x_idx = (tempy*KS) + tempx;
        Xd[i] += K[k_idx]*X[x_idx];
      }
    }
  }
  
  
  
  delete X;
  delete Xd;
}
