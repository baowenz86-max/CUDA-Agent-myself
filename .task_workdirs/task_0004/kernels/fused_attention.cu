#include <cuda_runtime.h>
#include <math.h>

__global__ void fused_attention_kernel(const float* x,const float* y,const float* iw,const float* ib,const float* ow,const float* ob,float* out,int B,int S,int R,int E,int H,float scale){
  int b=blockIdx.y,sq=blockIdx.x,e=threadIdx.x,Sr=S*R,D=E/H;
  if(b>=B||sq>=Sr||e>=E)return;
  extern __shared__ float sm[]; float* q=sm; float* p=sm+E; int srcq=sq/R;
  float qv=ib[e]; for(int j=0;j<E;++j) qv+=x[(b*S+srcq)*E+j]*iw[e*E+j]; q[e]=qv; __syncthreads();
  if(e==0){
    float mx=-CUDART_INF_F;
    for(int h=0;h<H;++h) for(int sk=0;sk<Sr;++sk){ int src=sk/R; float score=0;
      for(int d=0;d<D;++d){ int k=h*D+d; float kv=ib[E+k];
        for(int j=0;j<E;++j) kv+=y[(b*S+src)*E+j]*iw[(E+k)*E+j]; score+=q[k]*kv; }
      p[h*Sr+sk]=score*rsqrtf((float)D);
    }
    for(int h=0;h<H;++h){ float mx=-CUDART_INF_F,den=0; for(int sk=0;sk<Sr;++sk)mx=fmaxf(mx,p[h*Sr+sk]);
      for(int sk=0;sk<Sr;++sk){p[h*Sr+sk]=expf(p[h*Sr+sk]-mx);den+=p[h*Sr+sk];}
      for(int sk=0;sk<Sr;++sk)p[h*Sr+sk]/=den;
    }
  }
  __syncthreads();
  float result=ob[e];
  for(int k=0;k<E;++k){ int kh=k/D,kd=k%D; float c=0;
    for(int sk=0;sk<Sr;++sk){int src=sk/R,vd=kh*D+kd;float vv=ib[2*E+vd];
      for(int j=0;j<E;++j)vv+=y[(b*S+src)*E+j]*iw[(2*E+vd)*E+j]; c+=p[kh*Sr+sk]*vv;}
    result+=ow[e*E+k]*c;
  }
  out[(b*Sr+sq)*E+e]=tanhf(result*scale);
}
extern "C" void fused_attention_forward_launcher(const float* x,const float* y,const float* iw,const float* ib,const float* ow,const float* ob,float* out,int B,int S,int R,int E,int H,float scale,cudaStream_t stream){
  dim3 grid(S*R,B),block(E); size_t shared=(size_t)(E+H*S*R)*sizeof(float);
  fused_attention_kernel<<<grid,block,shared,stream>>>(x,y,iw,ib,ow,ob,out,B,S,R,E,H,scale);
}
