#!/bin/bash
i=1;
while [ $i -lt 1024 ]; do
	#if [[ -f ../floyd_war/floyd_warsh_cuda/logs_gtx_780_Ti/log$i/gpu_long_job.sh.o* ]];then
 		tm=`cat ../floyd_war/floyd_warsh_cuda/logs_gtx_780_Ti/log$i/gpu_long_job.sh.o* | grep "Time:" | awk '{print $2}'`;
		echo $i $tm >> ./gen_gts_780_Ti.dat;
	#fi
	echo $i;
	i=$(($i+1));
done
