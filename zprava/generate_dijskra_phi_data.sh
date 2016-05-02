#!/bin/bash
echo -n "" > dijskra_phi_data.dat;

i=1;
while [[ $i -lt 245 ]];do
	tm=`cat ../PAP/logs_phi/dijks_"$i" | grep "Duration:" | awk '{print $2}'`;
	echo "$i $tm" >> dijskra_phi_data.dat;
	i=$(($i+1));
done
