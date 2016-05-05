#!/bin/bash
i=1024;
mkdir log;
while [[ $i -gt 0 ]]; do
        sed -i -e "s/define BLOCK_SIZE 512/define BLOCK_SIZE $i/g" floyd.cu;
        nvcc floyd.cu -std=c++11 -O3 -o try;
        sed -i -e "s/define BLOCK_SIZE $i/define BLOCK_SIZE 512/g" floyd.cu;

        while [[ `qstat | grep beranm14` != "" ]]; do
                sleep 1;
                echo "waiting for $i";
        done

	qsub gpu_long_job.sh;

        while [[ `qstat | grep beranm14` != "" ]]; do
                sleep 1;
                echo "waiting for $i";
        done
	sleep 2;

	mv log logs_gtx_780_Ti/log$i;
	mkdir log
        i=$(($i-5));
done;
rm log;
