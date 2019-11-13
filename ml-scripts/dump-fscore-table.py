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
        return data["classifier"],data[subjects]

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

def process_results(result_dir):
    rows = []
    labels = ['none','sc_guard','oh_verify','cfi_register','cfi_verify']
    for perm in sorted(get_immediate_subdirectories(result_dir)):
        perm_dir = os.path.basename(perm)
        perm_result = os.path.join(perm,'result.json')
        re,subjects = read_result(perm_result)
        #print(re)
        classifier ={}
        for dic in re:
            protection, fscore = dic['label'],dic['fscore']
            classifier[protection]=fscore
        row = [perm_dir]
        for label in labels:
            if label not in classifier:
                classifier[label]='N/A'
            if label not in subjects:
                subjects[label]='N/A'
            #read fmeasure per program
            p_fmeasures, weights = read_fcsv(os.path.join(perm,'programs_fscore_{}.csv'.format(label)),label)
            weighted_stats = DescrStatsW(np.array(p_fmeasures).astype(float),weights=np.array(weights).astype(int),ddof=0)
            print(label,weighted_stats.mean,weighted_stats.std,weighted_stats.std_mean)
            row.append(classifier[label])
            row.append(subjects[label])
            row.append(weighted_stats.mean)
            row.append(weighted_stats.std)
            row.append(weighted_stats.std_mean)
        print(row)
        rows.append(row)

    with open('localization.csv', mode='w') as local_file:
        local_writer = csv.writer(local_file, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        rows.insert(0,['obfuscation','none','none_samples','none_median','none_std','none_std_mean','sc_guard','sc_samples','sc_median','sc_std','sc_std_mean','oh_verify','oh_vsamples','oh_median','oh_std','oh_std_mean','cfi_register','cfi_rsamples','cfir_median','cfir_std','cfir_std_mean','cfi_verify','cfi_vsamples','cfiv_median','cfiv_std','cfiv_std_mean'])
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
