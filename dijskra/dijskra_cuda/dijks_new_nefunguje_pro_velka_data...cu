#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib>
#include <limits>
#include <algorithm>

using namespace std;

const int BLOCK_SIZE = 512;


#define idx(i,j,lda) ( (j) + ((i)*(lda)) )

class mySet
{
private:
	int size = 4000;
	bool N[4000];
	int cnt = 4000;

	
public:
	__device__ mySet(){}
	
	
	__device__ void init(int s)
	{
		this->cnt = s;
		for (int i = 0; i < s; i++)
		{
			N[i] = true;
		}
	}

	__device__ bool contains(int x)
	{
		return N[x];
	}

	__device__ void insert(int x)
	{
		if (N[x] == true)
			return;
		N[x] = true;
		cnt++;
	}

	__device__ void erase(int x)
	{
		if (N[x] == true)
		{
			N[x] = false;
			cnt--;
		}
	}

	__device__ bool empty()
	{
		return (cnt == 0);
	}

	__device__ int getCount()
	{
		return cnt;
	}
};



__device__ int getGlobalIdx_1D_1D()
{
	return blockIdx.x *blockDim.x + threadIdx.x;
}

__device__ int getGlobalIdx_1D_2D()
{
	return blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
}

__device__ int getGlobalIdx_2D_2D()
{
	int blockId = blockIdx.x + blockIdx.y * gridDim.x;
	int threadId = blockId * (blockDim.x * blockDim.y)
		+ (threadIdx.y * blockDim.x) + threadIdx.x;
	return threadId;
}

__device__ int getGlobalIdx_2D_1D()
{
	int blockId   = blockIdx.y * gridDim.x + blockIdx.x;			 	
	int threadId = blockId * blockDim.x + threadIdx.x; 
	return threadId;
}
__device__ int getGlobalIdx_3D_3D()
{
	int blockId = blockIdx.x + blockIdx.y * gridDim.x
		+ gridDim.x * gridDim.y * blockIdx.z;
	int threadId = blockId * (blockDim.x * blockDim.y * blockDim.z)
		+ (threadIdx.z * (blockDim.x * blockDim.y))
		+ (threadIdx.y * blockDim.x) + threadIdx.x;
	return threadId;
}
//zdroj: http://cs.calvin.edu/courses/cs/374/CUDA/CUDA-Thread-Indexing-Cheatsheet.pdf


__global__ void prepareArray(int vertexCnt, int* d)
{
	int threads = gridDim.x * gridDim.y * gridDim.z * blockDim.x * blockDim.y * blockDim.z;
  	int cycleCnt = (vertexCnt / threads > 0 ? vertexCnt / threads : 1);
	for (int cycle = 0; cycle < cycleCnt; cycle++)
	{
		int s = (blockIdx.x * blockDim.x + threadIdx.x) + threads * cycle; // pozice na radku
		if(s >= vertexCnt) 
			return;

		for (int i = 0; i < vertexCnt; i++)
		{
			d[vertexCnt *i+s] = INT_MAX / 2;
		}
	}
}

__global__ void dijsktra( int* __restrict__ edgeMatrix, int vertexCnt, int* d)
{
    int blockX = gridDim.x * gridDim.y;//( vertexCnt + BLOCK_SIZE - 1 ) / BLOCK_SIZE;
	//int threadsX = gridDim.x * blockDim.x;
	//int threadsY = gridDim.y * blockDim.y;
  	__shared__ int cycleCnt;
 	int threadInBlockId = (threadIdx.y * blockDim.x) + threadIdx.x;	
  	if (threadInBlockId == 0)
    {
		cycleCnt = (vertexCnt / blockX > 0 ? vertexCnt / blockX : 1);
	}
    __syncthreads();
	for (int cycle = 0; cycle < cycleCnt; cycle++)
	{
		__shared__ int s; // (blockIdx.x * blockDim.x + threadIdx.x) + blockX * cycle; // pozice na radku
		
		__shared__ mySet N;
		__shared__ bool empty;
        if (threadInBlockId == 0)
        {
			s = blockIdx.x + blockIdx.y * gridDim.x + blockX * cycle;
			if (s < vertexCnt)
			{
				N.init(vertexCnt);
				empty = false;
				if (s*vertexCnt + s >= vertexCnt * vertexCnt)
					empty = true;
				d[s*vertexCnt + s] = 0;
			}
        }
        __syncthreads();
		if (s >= vertexCnt || empty == true)
			return;
		__shared__ int sharedMin[BLOCK_SIZE];
        __shared__ int sharedU[BLOCK_SIZE];
		while (!empty/*!N.empty()*/)
		{
			int localMin = INT_MAX;

			//int cnt = N.getCount();
			sharedMin[threadInBlockId] = INT_MAX;
            sharedU[threadInBlockId] = 0;
			int u = 0;
			int j = 0;
            int jobSize = (vertexCnt / BLOCK_SIZE > 0 ? vertexCnt / BLOCK_SIZE : 1);
			for (int i = threadInBlockId * jobSize; i < vertexCnt && j < jobSize; i++)
			{
                j++;
				if (!N.contains(i)) continue;
				if (localMin > d[vertexCnt *i+s])
				{
					localMin = d[vertexCnt *i+s];
					u = i;
				}
			}
            sharedMin[threadInBlockId] = localMin;
            sharedU[threadInBlockId] = u;
            __syncthreads();
            if (threadInBlockId == 0)
            {	
                for (int i = 1; i < BLOCK_SIZE; i++)
                {
                    if (localMin > sharedMin[i])
                    {
                        localMin = sharedMin[i];
                        u = sharedU[i];
                    }
                }
				sharedU[0] = u;
                N.erase(u);
            }
            __syncthreads();
            u = sharedU[0];
            int tempD = d[vertexCnt *u+s];
			j = 0;
			for (int i = threadInBlockId * jobSize; i < vertexCnt && j < jobSize; i++)
			{
				j++;
				int alt = edgeMatrix[u + i*vertexCnt];
				if (i == u || !N.contains(i) || alt <= 0) continue;

				alt += tempD;
				atomicMin((d + vertexCnt * i + s), alt);
				
			}
            
            if (threadInBlockId == 0)
            {
                empty = N.empty();
            }
            __syncthreads();
		}
		__syncthreads();
	}
}



int *readMatrix(const char *path, int &n)
{
	ifstream iFile;
	iFile.open(path);
	if (!iFile.is_open())
		return NULL;

	string line;
	getline(iFile, line);
	n = atoi(line.c_str());
	int *matrix = new int[n * n];
  
	int i;
	for (i = 0; getline(iFile, line); i++)
	{
		size_t endpos = line.find_last_not_of(" \t\r\n");
		if (string::npos != endpos)
			line = line.substr(0, endpos + 1);

		for (int j = 0; j < n; j++)
		{
			size_t pos = line.find_first_of(" ");
			matrix[idx(i,j,n)] = stoi(line, &pos, 10);
			line = line.substr(pos);
			
		}
	}
	iFile.close();
	return matrix;

}

void writeMatrix(const char *path, int n, int *matrix)
{
	ofstream oFile;
	oFile.open(path, fstream::trunc | fstream::out);
	if (oFile.is_open())
	{
		for (int i = 0; i < n; i++)
		{
			for (int j = 0; j < n; j++)
			{
				oFile << matrix[idx(i, j, n)] << (j + 1 < n ? " " : "\n");
			}
		}
	}
	oFile.close();
}

int main(int argc, const char* argv[])
{
	if (argc < 3 || argc > 4)
	{
		cout << "Program takes 2 or 3 parameters (matrix and thread count and optional output file)!\n";
		return 1;
	}

    int threadCnt = atoi(argv[2]);
	
    int stc = 0;
    int *matrix = readMatrix(argv[1], stc);
    // reading input file
    if (matrix == NULL){
         cout << "File doesn't exists" << endl;
         cout << argv[1] << endl;
         return 1;
    }
	cudaEvent_t start, stop;
	float elapsedTime;



	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);


	int *cumatrix; 
	int *d;
	cudaMalloc((void **)&cumatrix, stc * stc * sizeof(int));
	cudaMemcpy(cumatrix, matrix,  stc * stc * sizeof(int), cudaMemcpyHostToDevice);
	//int *lInts;

	//cudaHostAlloc( (void**)&lInts, stc*stc * sizeof(int), cudaHostAllocDefault ) ;
	//for (int i = 0; i < stc*stc;i++)
	//	lInts[i] = INT_MAX/2;
	cudaMalloc((void **)&d, stc * stc * sizeof(int));
	//cudaMemcpy(d, lInts,  stc * stc * sizeof(int), cudaMemcpyHostToDevice);
  

	dim3 dimGrid(( stc + BLOCK_SIZE - 1 ) / BLOCK_SIZE, stc);
	
	prepareArray<<<1, stc>>>(stc, d);
    cudaError_t code = cudaThreadSynchronize();  
	if (code != cudaSuccess)
	{
		fprintf(stdout, "GPUassert: %s \n", cudaGetErrorString(code));
	}
	

   	dijsktra<<<dimGrid, BLOCK_SIZE>>>(cumatrix, stc, d);
    code = cudaThreadSynchronize();  
	if (code != cudaSuccess)
	{
		fprintf(stdout, "GPUassert: %s \n", cudaGetErrorString(code));
	}
  
	int *outM = new int[stc*stc];

  	cudaMemcpy(outM, d,  stc * stc * sizeof(int), cudaMemcpyDeviceToHost);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	cout << "Time: " << elapsedTime << endl;
	if (argc == 4)
	{
		cout << "Writing results...\n";
		writeMatrix(argv[3], stc, outM);
	}
	cudaFree(cumatrix);
  	cudaFree(d);
  //	cudaFreeHost(lInts);

  	delete [] matrix;
  	delete [] outM;


	cout << "\nDone\n";
    return 0;
}
