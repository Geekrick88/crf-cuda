#!/bin/sh

learn_cmd="../../crf_learn"
test_cmd="../../crf_test"
eval_cmd="./conlleval.pl -d '\\t' < results-out.txt" 

if $# < 2
then
	$learn_options=""
fi

#../../crf_learn -a MIRA template train.data model
echo "running learning task"
../../build/crf_learn -c 4.0 template train.data.full model
#../../crf_test -m model test.data


#../../crf_test -m model test.data
echo "running test task"
#../../crf_learn -a CRF-L1 template train.data model
../../build/crf_test -m model test.data > results-out.txt

# rm -f model
echo "Converting output to Unix format"
dos2unix results-out.txt
echo "Generating Report"
./conlleval.pl -d '\t' < results-out.txt

