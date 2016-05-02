all:
	nvcc floyd.cu -std=c++11 -O3 -o try
profile:
	nvprof ./try ../../test_data/test4000x9.txt.rnd 4 output.txt
run:
	./try ../../test_data/test4000x9.txt.rnd 4 output.txt
