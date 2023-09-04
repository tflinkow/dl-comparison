import os
import pandas as pd
import numpy as np
import re
import csv

def process_reports(dataset, logic):
    folder_path = 'reports-lambda-search'
    rows = []

    reports = [f for f in os.listdir(folder_path) if (f'{dataset}_{logic}_' in f or f'{dataset}_baseline' in f) and f.endswith('.csv')]

    file_name = os.path.join('lambda-search', f'{dataset}-{logic}.csv')
    os.makedirs('lambda-search', exist_ok=True)

    with open(file_name, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(['Lambda', 'P-Acc', 'C-Acc'])

    best_combo = -1
    best_weight = -1

    for report in reports:
        file = os.path.join(folder_path, report)

        with open(file) as f:
            line = f.readline().strip()
            match = re.search(r'--dl-weight=([\d.]+)', line)
            assert match, f"Value not found in {file}."

            dl_weight = float(match.group(1))

        df = pd.read_csv(file, comment='#')

        p = df['Test-P-Acc'].values[-(len(df) // 10):]
        c = df['Test-C-Acc'].values[-(len(df) // 10):]
        s = p * c
        i = np.argmax(s)

        if (p[i] * c[i] > best_combo):
            best_combo = p[i] * c[i]
            best_weight = dl_weight

        rows.append([dl_weight, p[i], c[i]])

    rows.sort(key=lambda x: x[0])

    print(f"best weight for {dataset}_{logic} is {best_weight}")

    with open(file_name, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        for row in rows:
            writer.writerow(row)

implications = ['dl2', 'g', 'kd', 'lk', 'gg', 'rc', 'rc-s', 'rc-phi', 'yg']
conjunctions = ['dl2', 'g', 'lk', 'rc', 'yg' ]

configs = {
    'fmnist': implications,
    'cifar10': implications,
    'gtsrb': conjunctions
}

for dataset, logics in configs.items():
    for logic in logics:
        process_reports(dataset, logic)