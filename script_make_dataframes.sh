#python3 make_dataframes.py -d data -e EE -file_name /eos/user/r/rgargiul/www/dumphiggsdna/readyforcqrtrain/Run2022_Data_FG_corrscale.root
python3 make_dataframes.py -d mc -e EE -file_name /eos/user/r/rgargiul/www/dumphiggsdna/readyforcqrtrain/Run2022_MC_EFG_corrsmear.root --tree_name Event
#python3 make_dataframes.py -d data -e EB -file_name /eos/user/r/rgargiul/www/dumphiggsdna/readyforcqrtrain/Run2022_Data_FG_corrscale.root
python3 make_dataframes.py -d mc -e EB -file_name /eos/user/r/rgargiul/www/dumphiggsdna/readyforcqrtrain/Run2022_MC_EFG_corrsmear.root  --tree_name Event
