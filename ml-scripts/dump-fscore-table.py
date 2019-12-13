import json
import os
import csv
import numpy as np
from statsmodels.stats.weightstats import DescrStatsW
import sys

def get_immediate_subdirectories(d):
    return filter(os.path.isdir, [os.path.join(d, f) for f in os.listdir(d)])

def read_result(path):
    with open(path, 'r') as re_file:
        data = json.load(re_file)
        subjects = 'subject_groups'
        if subjects not in data:
            subjects = 'subject_groups_test' 
        return data["classifier"],data['Kfold_results'],data[subjects],int(data['train_size'])+int(data['test_size'])

def read_fcsv(csv_path, label):
    fmeasures,weights=[],[]
    if label in ['oh_verify','oh_hash']:
        label='OH'
    elif label in ['cfi_verify','cfi_register']:
        label='CFI'
    elif label in ['sc_guard']:
        label='SC'
    if not os.path.exists(csv_path):
        return [],[]
    with open(csv_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if row[0] in ['median','mean','std']:
                print(row[0])
                continue
            #binary name should have the protection label, otherwise that's an error
            if (label is not 'none' and label not in row[0]) or row[1]==0 or row[2]==0:
                print (label, row[0], row[1], row[2])
                continue
            fmeasures.append(row[1])
            weights.append(row[2])
        return fmeasures, weights
def process_folds(kfolds,labels):
    result = {}
    for label in labels:
        result[label]=[]
    for fold in kfolds:
        for scores in fold:
            if np.isnan(scores['fscore']):
                continue
            result[scores['label']].append(scores['fscore'])
    return result


def process_results(result_dir):
    rows = []
    labels = ['none','sc_guard','oh_verify','cfi_verify']
    for perm in sorted(get_immediate_subdirectories(result_dir)):
        perm_dir = os.path.basename(perm)
        perm_result = os.path.join(perm,'result.json')
        print('processing {}'.format(perm_result))
        re,kfolds,subjects,data_size = read_result(perm_result)
        row = [perm_dir.replace('sbb-','').replace('sbb','').replace('-','+').replace('FLAs','CFF').replace('BCF','BC').replace('SUB','IS'),data_size]
        fold_reads = process_folds(kfolds,labels)
        for label in labels:
            row.append(np.mean(fold_reads[label]))
            #row.append(subjects[label])
            row.append(np.std(fold_reads[label]))
        print(row)
        rows.append(row)

    with open(os.path.join(result_dir,'localization.csv'), mode='w') as local_file:
        local_writer = csv.writer(local_file, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        #rows.insert(0,['obfuscation','data_size','none','none_samples','none_std','sc_guard','sc_samples','sc_std','oh_verify','oh_vsamples','oh_std','cfi_verify','cfi_vsamples','cfiv_std'])
        rows.insert(0,['obfuscation','data_size','none','none_std','sc_guard','sc_std','oh_verify','oh_std','cfi_verify','cfiv_std'])
        for row in rows:
            local_writer.writerow(row)

def main():
    if len(sys.argv)<2:
        print('Provide the path to result files')
        exit(1)
    path = sys.argv[1]
    print('Reading result files located in {}'.format(path))
    process_results(path)



if __name__=='__main__':
    main()
