#include <iostream>
#include <iostream>
#include <stdio.h>
#include <ctype.h>
#include <set>
#include <string>
#include <stdlib.h>
#include <limits.h>

using namespace std;


int ** constructP(int ** d, int vertexCnt){
    int ** p = new int*[vertexCnt];
    for(int i=0;i<vertexCnt;i++){
        p[i] = new int[vertexCnt];
        for(int j=0;j<vertexCnt;j++){
            if (d[i][j] != 0 && d[i][j] != INT_MAX) {
                p[i][j] = i;
            } else {
                p[i][j] = -1;
            }
        }
    }
    return p;
}

void floydWars(int ** d, int vertexCnt){
    int ** p = constructP(d, vertexCnt);
    for(int i=0;i<vertexCnt;i++)
        for(int k=0;k<vertexCnt;k++)
            for(int j=0;j<vertexCnt;j++){
                d[i][j] = d[i][k] + d[k][j]
                p[i][j] = p[k][j]
            }
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
    int ** d = new int*[stc];
    for(int j=0;j<(int)stc;j++){
        d[j] = new int[stc];
        for(int i=0;i<(int)stc;i++){
            if(i == j){
                d[j][i] = 0;
            }
            else if (matrix[j][i] - '0' == 0)
                d[j][i] = INT_MAX;
            else
                d[j][i] = i;
            cout << " " << (d[j][i] == INT_MAX ? -1 : d[j][i]);
        }
        cout << endl;
    }
    // výpočet

    floydWars(d, stc);

    return 0;
}
