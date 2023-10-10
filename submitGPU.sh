#! /bin/bash

set -e

source /cvmfs/sft.cern.ch/lcg/app/releases/ROOT/6.24.08/x86_64-centos7-gcc48-opt/bin/thisroot.sh

begin=$(date +%s)

ispl=$1

EBEE=(EB EE)
nEvt=(10000 10000)
#nEvt=(3200000 970000) # for syst. uncertainty
if [ $2 -le 1 ]; then
  for i in ${!EBEE[@]}; 
  do
      python3 train_SS.py -e ${EBEE[i]} -n ${nEvt[i]} -s ${ispl}
      python3 train_SS_mc.py -e ${EBEE[i]} -n ${nEvt[i]} -r yes -s ${ispl} 
  done;
fi

if [ $2 -le 2 ]; then
  for EBEE in "EB" "EE"; 
  do 
      for data_type in "train"; #"test" 
      do
          echo correcting mc for ${EBEE} ${data_type}
          python3 correct_mc.py -e ${EBEE} -t ${data_type} -v 'SS' -s ${ispl}
      done
  done;
fi

if [ $2 -le 3 ]; then
  python3 train_preshower.py -s ${ispl}
  python3 train_preshower_mc.py -r yes -s ${ispl};
fi


EBEE=(EB EE)
nEvtI=(10000 10000)
if [ $2 -le 4 ]; then
  for i in ${!EBEE[@]};
  do
      echo "$i training for ${EBEE[i]} with n=${nEvtI[i]}"
      python3 train_Iso.py -e ${EBEE[i]} -n ${nEvtI[i]} -v Ph -s ${ispl}
      python3 train_Iso_mc.py -e ${EBEE[i]} -n ${nEvtI[i]} -v Ph -r yes -s ${ispl}
      python3 train_Iso.py -e ${EBEE[i]} -n ${nEvtI[i]} -v Ch -s ${ispl}
      python3 train_Iso_mc.py -e ${EBEE[i]} -n ${nEvtI[i]} -v Ch -r yes -s ${ispl}
  done;
fi

if [ $2 -le 5]; then
  for EBEE in "EB" "EE"; 
  do 
      for data_type in "train"; #"test" 
      do
          echo correcting mc for ${EBEE} ${data_type}
          python3 correct_mc.py -e ${EBEE} -t ${data_type} -v 'Iso' #-f yes -s ${ispl}
      done
  done;
fi

if [ $2 -le 6 ]; then
  EBEE=(EB EE)
  for i in ${!EBEE[@]}; 
  do
      python3 train_final_SS.py -e ${EBEE[i]} -n ${nEvt[i]} -s ${ispl} 
  done;
fi

if [ $2 -le 7 ]; then
  for i in ${!EBEE[@]};
  do
      python3 train_final_Iso.py -e ${EBEE[i]} -n ${nEvtI[i]} -v Ph -s ${ispl} 
      python3 train_final_Iso.py -e ${EBEE[i]} -n ${nEvtI[i]} -v Ch -s ${ispl} 
  done;
fi

if [ $2 -le 8 ]; then
  python3 train_final_preshower.py -s ${ispl};
fi

if [ $2 -le 9 ]; then
  for EBEE in "EB" "EE";
  do
      python3 correct_mc.py -e ${EBEE} -v "all" -f yes -s ${ispl}
      python3 correct_final.py -e ${EBEE} -v "all" -s ${ispl} 
      python3 correct_final_Iso.py -e ${EBEE} -v "all" -s ${ispl} 
  done;
fi

if [ $2 -le 10 ]; then
  EBEE=(EB EE)
  nEvtA=(7000000 2000000)
  for i in ${!EBEE[@]};
  do
      python3 check_results.py -e ${EBEE[i]} -n ${nEvtA[i]}
  done;
fi

tottime=$(echo "$(date +%s) - $begin" | bc)
echo ">>>>>>>>>>>>>>>>>>>> time spent: $tottime s"

