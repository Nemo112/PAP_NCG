#include <iostream>
#include <string>
#include <limits>
#include <fstream>
#include <algorithm>
#define BLOCK_SIZE 1007

using namespace std;


//floyd-warshall algorithm
//finds shortest paths to every vertex in matrix for all vertexes
/*__global__ void floyd(int *matrix, int l)
{
	//int LARGE_INT = numeric_limits<int>::max() / 2; //for our purposes practically infinity
	int c = threadIdx.x;
	const int n = l;
	int i, j, k = 0;
 	int my_start = (c  ) * n / M;
  	int my_end   = (c+1) * n / M;

	for (int i = my_start; i < my_end; i++)
	{
		for (int j = 0; j < n; j++)
		{
			if (i == j)
				matrix[i*n+j] = 0;
			else if (matrix[i*n+j] == 0)
				matrix[i*n+j] = INT_MAX/2; // 0xDFFF;
		}
	}

	for (k = 0; k < n; k++)
	{
		for (i = my_start; i < my_end; i++)
		{
			for (j = 0; j < n; j++)
			{
				matrix[i*n+j] = min(matrix[i*n+j], matrix[i*n+k] + matrix[k*n+j]);
			}
		}
		__syncthreads(); 
	}

}*/

//Reads incidence matrix.
//Matrix has to look like this (including spaces):
//1 5 9 0 ... 4
//5 0 11 2 ...0
//...
//4 .....
//Where 0 means vertex is not a neighbour and > 0 values equal distance
//int &n is the size of the matrix (specified on the 1st line of the input file)
int *readMatrix(const char *path, int &n)
{
	ifstream iFile;
	iFile.open(path);
	if (!iFile.is_open())
		return NULL;

	string line;
	getline(iFile, line);
	n = atoi(line.c_str());
	int *matrix = new int[n*n];
	int i;
	for (i = 0; getline(iFile, line); i++)
	{
		size_t endpos = line.find_last_not_of(" \t\r\n");
		if (string::npos != endpos)
			line = line.substr(0, endpos + 1);
		//matrix[i] = new int[n];
		for (int j = 0; j < n; j++)
		{
			size_t pos = line.find_first_of(" ");
			matrix[i*n+j] = stoi(line, &pos, 10);
			line = line.substr(pos);
		}
	}
	iFile.close();
	return matrix;

}

//Writes matrix to file at path
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
				oFile << matrix[i*n+j] << (j + 1 < n ? " " : "\n");
			}
		}
	}
	oFile.close();
}

void printMatrix(int *matrix, int n)
{
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
		if (matrix[i*n+j] == INT_MAX)
			cout << "0 ";
		else
			cout << matrix[i*n+j] << " ";
		cout << endl;
	}
}

__global__ void floyd_prepare(int *matrix, int n){
  int column = blockIdx.x * blockDim.x + threadIdx.x;
  int idx = n * blockIdx.y + column;
  if(column >= n )
      return;
  if ( blockIdx.y == column)
    matrix[idx] = 0;
  else if(matrix[idx] == 0)
    matrix[idx] = INT_MAX/2;
}

__global__ void floyd_kern(int k, int *matrix, int n){
  int column = blockIdx.x * blockDim.x + threadIdx.x; // pozice na radku
  if(column >= n ) // pokud přepluje, pravděpodobně jsme přejeli velikost bloku
      return;
  int idx = n * blockIdx.y + column; // přesná pozice v matici
  
  __shared__ int bmatch; // pro jedno k nejlepší hodnota v floyd
  
  if(threadIdx.x == 0)
      bmatch = matrix[n*blockIdx.y + k];
  
  __syncthreads();
  
  if(bmatch == INT_MAX/2)
     return;
  
  int tmp = matrix[k*n+column];
  
  if(tmp == INT_MAX/2) 
    return;
  
  int current = bmatch + tmp;
  if(current < matrix[idx]){
    matrix[idx] = current; // matrix[i*n+j] = min(matrix[i*n+j], matrix[i*n+k] + matrix[k*n+j]);
  }
  /*
   Není to moc SIMT, ale jako merge slajdů z 
   https://edux.fit.cvut.cz/courses/MI-PAP/_media/lectures/zaklady.pdf
   https://edux.fit.cvut.cz/courses/MI-PAP/_media/lectures/zaklady2.pdf
   https://edux.fit.cvut.cz/courses/MI-PAP/_media/lectures/zaklady3.pdf
   to docela jede. 
   */
}

void gpu_floyd(int *matrix, int n){
  int *cumatrix; 
  cudaMalloc((void **)&cumatrix, n * n * sizeof(int));
  cudaMemcpy(cumatrix, matrix,  n * n * sizeof(int), cudaMemcpyHostToDevice);
  
  dim3 dimGrid(( n + BLOCK_SIZE - 1 ) / BLOCK_SIZE, n);
  
  floyd_prepare<<<dimGrid, BLOCK_SIZE>>>(cumatrix, n);
  
  cudaThreadSynchronize();
     
  for (int k = 0; k < n; k++){
    floyd_kern<<<dimGrid, BLOCK_SIZE>>>(k, cumatrix, n);
    cudaThreadSynchronize();  
  }
  
  cudaMemcpy(matrix, cumatrix, sizeof(int)*n*n,cudaMemcpyDeviceToHost);  
  cudaFree(cumatrix);
}

int main(int argc, const char* argv[])
{
	if (argc < 3 || argc > 4)
	{
		cout << "Program takes 2 or 3 parameters (matrix and thread count and optional output file)!\n";
		return 1;
	}

	int n = 0;
	int *matrix = readMatrix(argv[1], n);
	
/*	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
		{
			if (i == j)
				matrix[i*n+j] = 0;
			else if (matrix[i*n+j] == 0)
				matrix[i*n+j] = INT_MAX/2; // 0xDFFF;
		}
	}*/
	

	if (matrix == NULL)
	{
		cout << "Couldn't read matrix!\n";
		return 2;
	}

	//printMatrix(matrix, n);

	cudaEvent_t start, stop;
	float elapsedTime;


	cudaEventCreate( &start ) ; 
	cudaEventCreate( &stop ) ; 
	cudaEventRecord( start, 0 );

	
	gpu_floyd(matrix, n);

	cudaEventRecord( stop, 0 );
	cudaEventSynchronize( stop ) ; 
	cudaEventElapsedTime( &elapsedTime, start, stop );	
	

	cout << "Time: " << elapsedTime << endl;	
	//printMatrix(matrix, n);

	if (argc == 4)
	{
		cout << "Writing results...\n";
		writeMatrix(argv[3], n, matrix);
	}

	delete[] matrix;
	cout << "\nDone\n";
	return 0;
}
