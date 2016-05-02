#include <iostream>
#include <string>
#include <limits>
#include <fstream>
#include <algorithm>
#include <omp.h>

using namespace std;

int LARGE_INT = numeric_limits<int>::max() / 2; //for our purposes practically infinity

//floyd-warshall algorithm
//finds shortest paths to every vertex in matrix for all vertexes
void floyd(int **matrix, int n, int threadCnt)
{
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
		{
			if (i == j)
				matrix[i][j] = 0;
			else if (matrix[i][j] == 0)
				matrix[i][j] = LARGE_INT;
		}
	}
	if (threadCnt > 0)
		omp_set_num_threads(threadCnt);

	int i, j, k;
	for (k = 0; k < n; k++)
	{
		#pragma omp parallel for private(i,j)
		for (i = 0; i < n; i++)
		{
			for (j = 0; j < n; j++)
			{
			    matrix[i][j] = min(matrix[i][j], matrix[i][k] + matrix[k][j]);
				/*if (matrix[i][j] > matrix[i][k] + matrix[k][j])
				{
					matrix[i][j] = matrix[i][k] + matrix[k][j];
				}*/
			}
		}
	}

}

//Reads incidence matrix.
//Matrix has to look like this (including spaces):
//1 5 9 0 ... 4
//5 0 11 2 ...0
//...
//4 .....
//Where 0 means vertex is not a neighbour and > 0 values equal distance
//int &n is the size of the matrix (specified on the 1st line of the input file)
int **readMatrix(const char *path, int &n)
{
	ifstream iFile;
	iFile.open(path);
	if (!iFile.is_open())
		return NULL;

	string line;
	getline(iFile, line);
	n = atoi(line.c_str());
	int **matrix = new int*[n];
	int i;
	for (i = 0; getline(iFile, line); i++)
	{
		size_t endpos = line.find_last_not_of(" \t\r\n");
		if (string::npos != endpos)
			line = line.substr(0, endpos + 1);
		matrix[i] = new int[n];

		for (int j = 0; j < n; j++)
		{
			size_t pos = line.find_first_of(" ");
			matrix[i][j] = stoi(line, &pos, 10);
			line = line.substr(pos);
		}
	}
	iFile.close();
	return matrix;

}

//Writes matrix to file at path
void writeMatrix(const char *path, int n, int **matrix)
{
	ofstream oFile;
	oFile.open(path, fstream::trunc | fstream::out);
	if (oFile.is_open())
	{
		for (int i = 0; i < n; i++)
		{
			for (int j = 0; j < n; j++)
			{
				oFile << matrix[i][j] << (j + 1 < n ? " " : "\n");
			}
		}
	}
	oFile.close();
}

void printMatrix(int **matrix, int n)
{
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
		if (matrix[i][j] == LARGE_INT)
			cout << "0 ";
		else
			cout << matrix[i][j] << " ";
		cout << endl;
	}
}

int main(int argc, const char* argv[])
{
	if (argc < 3 || argc > 4)
	{
		cout << "Program takes 2 or 3 parameters (matrix and thread count and optional output file)!\n";
		return 1;
	}

	int n = -1;
	int **matrix = readMatrix(argv[1], n);

	if (matrix == NULL)
	{
		cout << "Couldn't read matrix!\n";
		return 2;
	}

	//printMatrix(matrix, n);
	int threadCnt = atoi(argv[2]);

	double startTime = omp_get_wtime();


	floyd(matrix, n, threadCnt); //perform floyd-warshall algorithm

	double stopTime = omp_get_wtime() - startTime;
	cout << "\n\n------------------\n\n";
	//printMatrix(matrix, n);

	cout << "\nDuration: " << stopTime << endl;

	if (argc == 4)
	{
		cout << "Writing results...\n";
		writeMatrix(argv[3], n, matrix);
	}

	for (int i = 0; i < n; i++)
		delete[] matrix[i];
	delete[] matrix;
	cout << "\nDone\n";
	return 0;
}
