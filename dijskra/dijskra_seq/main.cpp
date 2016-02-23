#include <iostream>
#include <iostream>
#include <stdio.h>
#include <ctype.h>
#include <set>
#include <string>
#include <stdlib.h>

using namespace std;

int * dijsktra(string * edgeMatrix, int vertexCnt,int s){
    int * p = new int[vertexCnt];
    int * d = new int[vertexCnt];
    set <int> * N = new set<int>;
    for(int i = 0; i < vertexCnt; i++){
        p[i] = vertexCnt + 1; // infinity
        d[i] = -1; // uknown
        N->insert(i);
    }
    d[s] = 0;
    while (!N->empty()){
        int u = *N->begin();
        N->erase(u);
        for(int v = 0; v < vertexCnt; v ++)
            if(edgeMatrix[u][v] == '1'){
                int alt = (d[u] == -1 ? 1 : d[u] + 1); // l(u,v), kde l je vždy 1
                //cout << "Alt: " << alt << " d v " << d[v] << endl;
                if (alt < d[v] || d[v] == -1){
                    d[v] = alt;
                    p[v] = u;
                }
            }
    }
    return d;
}

int main(int argc, const char* argv[])
{
    // input tests
    FILE *fp = NULL;
    if (argc < 2){
         cout << "Give name of relationship matrix file" << endl;
         return 2;
    }
    fp = fopen(argv[argc-1],"r");
    if (fp == NULL){
         cout << "File doesn't exists" << endl;
         cout << argv[argc-2] << endl;
         return 1;
    }
    // read variables
    char ch;
    unsigned int stc; // count of vertexes
    int i = 0;
    bool frst = false;
    string stsCnt;
    string * matrix;
    // reading input file
    ch = fgetc(fp);
    while(ch != EOF ){
        if(ch == '\n'){
            if(frst == false){
                stc=atoi(stsCnt.c_str());
                frst = true;
                matrix = new string[stc];
            }else
                i ++;
        }else{
            if (frst == false)
                stsCnt.push_back(ch);
            else
                matrix[i].push_back(ch);
        }
        ch = fgetc(fp);
    }
    fclose(fp);
    // display
    for(unsigned int j=0;j<stc;j++){
        cout << matrix[j] << endl;
    }
    // Výpočet
    for(int j=0;j<(int)stc;j++){
        int * d = dijsktra(matrix, stc, j);
        for(int i=0;i<(int)stc;i++){
            cout << d[i] << " ";
        }
        cout << endl;
        delete [] d;
    }

    return 0;
}
