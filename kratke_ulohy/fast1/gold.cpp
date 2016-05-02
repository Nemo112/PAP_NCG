/*  Short job 1
*/ 
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <omp.h>
#include <xmmintrin.h>


void testuj(float *mX,float *mX2,int n,int m)
{
  // test spravnosti reseni
  // test for correctness
  int i,j,k,pk,err;
  float pom,k1;
  err=0;
  for(i=0;i<(n*m);i++)
  {    
    if (fabs(mX[i]-mX2[i])>0.1)  
    {
      err++;
      printf("%i = %g,%g\n",i,mX[i],mX2[i]);
    }
  }
  if (err!=0) printf("total ERR=%i/%i \n",err,(n*m));
}
 
  
int vyprazdni(int *temp,int k)
{
  // vyprazdni cache
  // flush cache
  int i,s=0;
  for(i=0;i<k;i++)
      s+=temp[i];
  return s;
}  


//!! zacatek casti k modifikaci
//!! beginning of part for modification
void Gauss_BS( const float * __restrict__  inA, const float * __restrict__  inB, float * outX, const int tn, const int tm) {
  int i,i2,j,l=0,k=0,chunk;
  float s;
  const float * pa;
  const float * pb;
  omp_set_num_threads (4);
  #pragma omp parallel for private(i,j,s,pa,pb) schedule (runtime)
  for(k=0;k<tm;k++){
   for(i=tn-1;i>=0;i--){
      i2=(tn-1)-i;
      pa = &inA[i*tn+(i+1)];
      pb = &outX[(i+1)*tm+k]; 
      s = inB[i*tm+k];
      //for(j=i+1;j<tn;j++){
      for(j=0;j<i2;j++){
	//s -= inA[i*tn+j]*outX[j*tm+k];
	s -= *(pa) * *(pb);
	pa ++;
	pb += tm;
      }
      outX[i*tm+k]=s/inA[i*tn+i];
    }
  }
}
//!! end of part for modification
//!! konec casti k modifikaci

void  printM(float * mX,int n,int m){
  int i=0, j=0;
   for(int i=0; i < n; i++){
    for(int j=0; j < m; j++){
      printf("%f ",mX[i*m+j]); 
    } 
    printf("\n");
  }
  
}

int main( void ) {

  double start_time,end_time,timea[10];
  
  int soucet=0,N,i,j,k,n,m,*pomo,v;
  int ri,rj,rk;
  double delta,s_delta=0.0;
  float *mA, *mB,*mX,*mX2,s; 
    
  //int tn[4]={1000,1500,2000,2500};  
  int tn[4]={1*1024,2*1024,12*1024,16*1024};
  int tm[4]={1024,256,32,12};
  srand (time(NULL));   
  pomo=(int *)malloc(32*1024*1024);    
  v=0;    
  
  for(N=0;N<10;N++)
     timea[N]=0.0;
  
  for(N=0;N<4;N++){
    n=tn[N];
    m=tm[N];

    mA=(float *)malloc(n * n * sizeof(float));
    mB=(float *)malloc(n * m * sizeof(float));
    mX=(float *)malloc(n * m *sizeof(float));
    mX2=(float *)malloc(n * m * sizeof(float));
    if ((mA==NULL)||(mB==NULL)||(mX==NULL)||(mX2==NULL)){
      printf("Insufficient memory!\n"); 
      return -1;
    }
    
    for (i=0; i<n; i++) {
      for (j=0; j<n; j++) {
	if (j>=i)
	  mA[i*n+j] = (float)(2*(rand()%59)-59);
	else  
	  mA[i*n+j] = 0.0;
      }
    }
    for (k=0; k<m; k++) {
      for (j=0; j<n; j++) {  
	mX2[j*m+k] = (float)((rand()%29)-14);     
      }
    }
    
    for (k=0; k<m; k++) {
      for (i=0; i<n; i++) {
	s=0.0;
	for (j=0; j<n; j++) {
	  s += mA[i*n+j]*(mX2[j*m+k]);
	}
	mB[i*m+k]=s;
      }
    }
    soucet+=vyprazdni(pomo,v);
    start_time=omp_get_wtime();
    // improve performance of this call
    // vylepsit vykonnost tohoto volani
    Gauss_BS( mA, mB, mX, n, m);
    //printM(mX,n,m);
			    
    end_time=omp_get_wtime();
    delta=end_time-start_time;
    timea[0]+=delta;
    testuj(mX,mX2,n,m);
    printf("n0=%i m0=%i time=%g \n",n,m,delta);        
    fflush(stdout);
  
    free(mX2);
    free(mX);
    free(mB);
    free(mA);
  } 
  printf("%i\n",soucet); 
  printf("Total time=%g\n",timea[0]);
  //for(N=0;N<10;N++)  printf("%i=%g\n",N,timea[N]); 

  return 0;
}
