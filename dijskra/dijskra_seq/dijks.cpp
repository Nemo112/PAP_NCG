#include <iostream>
#include <fstream>
#include <set>
#include <string>
#include <cstdlib>
#include <omp.h>
#include <limits>
#include <algorithm>

using namespace std;

int LARGE_INT = numeric_limits<int>::max() / 2;


int * dijsktra(int ** edgeMatrix, int vertexCnt, int s){
	int * p = new int[vertexCnt];
	int * d = new int[vertexCnt];
	set <int> * N = new set<int>;
	for (int i = 0; i < vertexCnt; i++){
		d[i] = LARGE_INT; // infinity
		p[i] = -1; // uknown
		N->insert(i);
	}
	d[s] = 0;
	int next = s;
	while (!N->empty()){
		int u = next;
		N->erase(u);
		next = *N->begin();

		set<int>::iterator it;
		for (it = N->begin(); it != N->end(); ++it)
		{
			if (edgeMatrix[u][*it] > 0) //zrusit if (velka hodnota uzlu)?
			{
				int alt = d[u] + edgeMatrix[u][*it];
				d[*it] = min(alt, d[*it]);
				if (alt < d[*it]){
					p[*it] = u;
				}
			}
			if (d[next] > d[*it])
				next = *it;
		}
	}
	return d;
}

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


int main(int argc, const char* argv[])
{

	if (argc < 3 || argc > 4)
	{
		cout << "Program takes 2 or 3 parameters (matrix and thread count and optional output file)!\n";
		return 1;
	}

    // read variables
    int stc; // count of vertexes
    int threadCnt = atoi(argv[2]);
    int **matrix = readMatrix(argv[1], stc);
    // reading input file
    if (matrix == NULL){
         cout << "File doesn't exists" << endl;
         cout << argv[1] << endl;
         return 1;
    }
    // display
  /*  for (int i = 0; i<stc;i++) {
        for(int j=0;j<stc;j++){
            cout << matrix[i][j] << " ";
        }
        cout << endl;
    }
    cout << "\n\n------\n\n";*/
    // Výpoèet



    double startTime = omp_get_wtime();

	if (threadCnt > 0)
		omp_set_num_threads(threadCnt);
    //zbyva paralelizovat
    //neco jsem zkousel, ale zrychleni bylo bidny,
    //pristi tyden si nad to poradne sednu a dodelam to
	int j;
	int **d = new int*[stc];
	//#pragma omp parallel for private(j)
    for(j = 0; j < stc; j++){
        d[j] = dijsktra(matrix, stc, j);
    }
    double stopTime = omp_get_wtime() - startTime;

    cout << "\nDuration: " << stopTime << endl;

	if (argc == 4)
	{
		cout << "Writing results...\n";
		writeMatrix(argv[3], stc, d);
	}


	for (int i = 0; i < stc; i++)
		delete[] d[i];
	delete[] d;

	cout << "\nDone\n";
    return 0;
}
