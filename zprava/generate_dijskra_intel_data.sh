#!/bin/bash
echo -n "" > dijskra_xeon_data.dat;

i=1;
while [[ $i -lt 25 ]];do
	tm=`cat ../PAP/scripts/djsk_n"$i"logs/gpu_long_job.sh.o* | grep "Duration:" | awk '{print $2}'`;
	echo "$i $tm" >> dijskra_xeon_data.dat;
	i=$(($i+1));
done
