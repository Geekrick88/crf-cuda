#!/bin/sh
../../build/crf_learn -c 10.0 template train.data model
../../build/crf_test -m model test.data > results-out.txt


rm -f model

echo "Converting output to Unix format"
dos2unix results-out.txt

echo "Generating Report"
./conlleval.pl -d '\t' < results-out.txt