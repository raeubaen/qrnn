#! /bin/bash

set -e

source /cvmfs/sft.cern.ch/lcg/app/releases/ROOT/6.24.08/x86_64-centos7-gcc48-opt/bin/thisroot.sh

begin=$(date +%s)

ispl=$1

if [ $2 -le 1 ]; then
  source script_weightuniform.sh
  source script_transform.sh
fi

EBEE=(EB EE)
nEvt=(500000 500000)
if [ $2 -le 2 ]; then
  for i in ${!EBEE[@]}; 
  do
      python3 train_SS.py -e ${EBEE[i]} -n ${nEvt[i]} -s ${ispl}
  done;
fi

EBEE=(EB EE)
nEvt=(500000 500000)
if [ $2 -le 3 ]; then
  for i in ${!EBEE[@]};
  do
      python3 train_SS_mc.py -e ${EBEE[i]} -n ${nEvt[i]} -r yes -s ${ispl}
  done;
fi

if [ $2 -le 4 ]; then
  for EBEE in "EB" "EE"; 
  do 
      for data_type in "train"; #"test" 
      do
          echo correcting mc for ${EBEE} ${data_type}
          python3 correct_mc.py -e ${EBEE} -t ${data_type} -v 'SS' -s ${ispl}
      done
  done;
fi

tottime=$(echo "$(date +%s) - $begin" | bc)
echo ">>>>>>>>>>>>>>>>>>>> time spent: $tottime s"


