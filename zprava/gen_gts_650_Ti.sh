#!/bin/bash
i=1;
while [ $i -lt 1024 ]; do
	if [[ ../floyd_war/floyd_warsh_cuda/logs_gtx_650_Ti/log$i.txt ]];then
 		tm=`cat ../floyd_war/floyd_warsh_cuda/logs_gtx_650_Ti/log$i.txt | grep "Time:" | awk '{print $2}'` ;
		echo $i $tm >> gen_gts_650_Ti.dat;
	fi
	i=$(($i+1));
done
