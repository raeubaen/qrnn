import os
import argparse
import uproot
import pandas as pd
import numpy as np

def get_norm(df, part):
    return np.sqrt(df[f"{part}_energyRaw"]**2 - df[f"{part}_px"]**2 - df[f"{part}_py"]**2 - df[f"{part}_pz"]**2)

def make_dataframe(path, tree, data_key, EBEE, dfDir, dfname, cut=None, split=None, sys=False):

    Branches = ['probe_eta', 'probe_esEffSigmaRR', 'tag_pfChargedIsoPFPV',
                'tag_r9', 'tag_phiWidth', 'probe_pt', 'tag_esEffSigmaRR',
                'tag_phi', 'probe_energyRaw',
                'tag_pfPhoIso03', 'tag_eta',
                'tag_pt', 'tag_s4', 'tag_sieie',
                'tag_sipip', 'tag_sieip', 'tag_energyRaw',
                'tag_pfChargedIsoWorstVtx', 'probe_phi', 'probe_sipip',
                'tag_etaWidth', 'probe_hoe',
                'probe_pfRelIso03_all_Fall17V2', 'probe_electronVeto', 'probe_pfPhoIso03']

    variables = ['probe_etaWidth', 'probe_r9', 'probe_s4',
                'probe_phiWidth', 'probe_sieie',
                'probe_sieip', 'probe_pfChargedIsoPFPV',
                'probe_pfChargedIsoWorstVtx', 'probe_esEnergyOverRawE']

    rename_dict = {'probe_eta': 'probeScEta', 'probe_esEffSigmaRR': 'probeSigmaRR', 'tag_pfChargedIsoPFPV': 'tagChIso03', 'tag_r9': 
    'tagR9', 'tag_phiWidth': 'tagPhiWidth', 'probe_pt': 'probePt', 'tag_esEffSigmaRR': 'tagSigmaRR', 'tag_phi': 'tagPhi', 
    'probe_energyRaw': 'probeScEnergy', 'tag_pfPhoIsoPFPV': 'tagPhoIso', 'tag_eta': 'tagScEta', 'tag_pt': 'tagPt', 'tag_s4': 'tagS4', 
    'tag_sieie': 'tagSigmaIeIe', 'tag_sipip': 'tagCovarianceIpIp', 'tag_sieip': 'tagCovarianceIeIp', 'tag_energyRaw': 'tagScEnergy', 
    'tag_pfChargedIsoWorstVtx': 'tagChIso03worst', 'probe_phi': 'probePhi', 'probe_sipip': 'probeCovarianceIpIp', 'tag_etaWidth': 
    'tagEtaWidth', 'probe_hoe': 'probeHoE', 'probe_pfRelIso03_all_Fall17V2': 'probeNeutIso', 'probe_electronVeto': 
    'probePassEleVeto', 'probe_etaWidth': 'probeEtaWidth', 'probe_r9': 'probeR9', 'probe_s4': 'probeS4', 'probe_phiWidth': 
    'probePhiWidth', 'probe_sieie': 'probeSigmaIeIe', 'probe_sieip': 'probeCovarianceIeIp', 'probe_pfChargedIsoPFPV': 
    'probeChIso03', 'probe_pfChargedIsoWorstVtx': 'probeChIso03worst', 'probe_esEnergyOverRawE': 'probeesEnergyOverSCRawEnergy',
    'probePhoIso': 'probe_pfPhoIso03'} 

    branches = Branches + variables

    ptmin = 25.
    ptmax = 150.
    etamin = -2.5
    etamax = 2.5
    phimin = -3.14
    phimax = 3.14
    
    
    print(f'load root files from {path}, tree name: {tree}')
    root_file = uproot.open(path)
    up_tree = root_file[tree]

    print(branches)

    df = up_tree.arrays(branches, library='pd')
    print(df.keys())
    for p in ["tag", "probe"]:
      df[f"{p}_theta"] = 2*np.arctan(np.exp(-df[f"{p}_eta"]))
      df[f"{p}_px"] = df[f"{p}_energyRaw"] * np.sin(df[f"{p}_theta"]) * np.cos(df[f"{p}_phi"])
      df[f"{p}_py"] = df[f"{p}_energyRaw"] * np.sin(df[f"{p}_theta"]) * np.sin(df[f"{p}_phi"])
      df[f"{p}_pz"] = df[f"{p}_energyRaw"] * np.cos(df[f"{p}_theta"])
    for v in ["energyRaw", "px", "py", "pz"]:
      df[f"dilepton_{v}"] = df[f"tag_{v}"] + df[f"probe_{v}"]
    df["mass"] = get_norm(df, "dilepton")
    df["rho"] = df.mass

    print('renaming data frame columns: ', rename_dict)
    df.rename(columns=rename_dict, inplace=True)
    print(df.keys())
    print(df)
    
    df.query('probePt>@ptmin and probePt<@ptmax and probeScEta>@etamin and probeScEta<@etamax and probePhi>@phimin and probePhi<@phimax',inplace=True)
    
    if EBEE == 'EB': 
        df.query('probeScEta>-1.4442 and probeScEta<1.4442',inplace=True)
    elif EBEE == 'EE': 
        df.query('probeScEta<-1.556 or probeScEta>1.556',inplace=True)
    
    if cut is not None: 
        print('apply additional cut: ', cut)
        df.query(cut,inplace=True)
    
    df = df.sample(frac=1.).reset_index(drop=True)
    
    if sys: 
        df_train1 = df[0:int(0.45*df.index.size)]
        df_train2 = df[int(0.45*df.index.size):int(0.9*df.index.size)]
        df_test = df[int(0.9*df.index.size):]
        df_train1.to_hdf('{}/{}_train_split1.h5'.format(dfDir,dfname),'df',mode='w',format='t')
        df_train2.to_hdf('{}/{}_train_split2.h5'.format(dfDir,dfname),'df',mode='w',format='t')
        df_test.to_hdf('{}/{}_test.h5'.format(dfDir,dfname),'df',mode='w',format='t')
        print('{}/{}_(train/test).h5 have been created'.format(dfDir,dfname))
    else:
        if split is not None: 
            df_train = df[0:int(split*df.index.size)]
            df_test = df[int(split*df.index.size):]
            df_train.to_hdf('{}/{}_train.h5'.format(dfDir,dfname),'df',mode='w',format='t')
            df_test.to_hdf('{}/{}_test.h5'.format(dfDir,dfname),'df',mode='w',format='t')
            print('{}/{}_(train/test).h5 have been created'.format(dfDir,dfname))
        else: 
            df.to_hdf('{}/{}.h5'.format(dfDir,dfname),'df',mode='w',format='t')
            print('{}/{}.h5 have been created'.format(dfDir,dfname))


def main(options):

    path = {'data': options.file_name,
              'mc': options.file_name}
    tree = {'data': options.tree_name,
              'mc': options.tree_name}

    cut = 'tagPt>40 and tagR9>0.8 and mass>80 and mass<100 and probeChIso03<6 and tagScEta>-2.1 and tagScEta<2.1 and probePassEleVeto==0'

    cutIso = {'EB': 'tagPt>40 and tagR9>0.8 and mass>80 and mass<100 and probeSigmaIeIe<0.0105 and tagScEta>-2.1 and tagScEta<2.1 and probePassEleVeto==0',
              'EE': 'tagPt>40 and tagR9>0.8 and mass>80 and mass<100 and probeSigmaIeIe<0.028 and tagScEta>-2.1 and tagScEta<2.1 and probePassEleVeto==0'}

    cutplots = 'tagPt>40 and probePt>20 and mass>80 and mass<100 and probePassEleVeto==0 and tagScEta<2.5 and tagScEta>-2.5' 
    
    data_key = options.data_key
    EBEE = options.EBEE 
    split = 0.9
    dfDir = f'./tmp_dfs/sys'

    if not os.path.exists(dfDir): 
        os.makedirs(dfDir)

    make_dataframe(options.file_name, options.tree_name, data_key, EBEE, dfDir, 'df_{}_{}'.format(data_key, EBEE), cut, sys=True)
    make_dataframe(options.file_name, options.tree_name, data_key, EBEE, dfDir, 'df_{}_{}_Iso'.format(data_key, EBEE), cutIso[EBEE], sys=True)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    requiredArgs = parser.add_argument_group('Required Arguements')
    requiredArgs.add_argument('-d','--data_key', action='store', type=str, required=True)
    requiredArgs.add_argument('-e','--EBEE', action='store', type=str, required=True)
    requiredArgs.add_argument('-file_name','--file_name', action='store', type=str, required=True)
    requiredArgs.add_argument('-tree_name','--tree_name', action='store', type=str, default="tree")

    options = parser.parse_args()
    main(options)
