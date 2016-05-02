#!/bin/bash
echo -n "" > floyd_xeon_data.dat;

i=1;
while [[ $i -lt 25 ]];do
	tm=`cat ../PAP/scripts/floyd_n"$i"logs/gpu_long_job.sh.o* | grep "Duration:" | awk '{print $2}'`;
	echo "$i $tm" >> floyd_xeon_data.dat;
	i=$(($i+1));
done
