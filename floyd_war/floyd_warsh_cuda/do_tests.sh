#!/bin/bash

i=1024;

while [[ $i -gt 0 ]]; do
	sed -i -e "s/define BLOCK_SIZE 512/define BLOCK_SIZE $i/g" floyd.cu;
	nvcc floyd.cu -std=c++11 -o try;
	sed -i -e "s/define BLOCK_SIZE $i/define BLOCK_SIZE 512/g" floyd.cu;
	./try ../../test_data/test4000x9.txt.rnd 4 output.txt > logs_gtx_650_Ti/log$i.txt;
	i=$(($i-1));
done;
