import os
import pandas as pd
import numpy as np
import re

folder_path = 'reports'

experiments = {
    'fmnist': [f for f in os.listdir(folder_path) if 'fmnist' in f and f.endswith('.csv')],
    'cifar10': [f for f in os.listdir(folder_path) if 'cifar10' in f and f.endswith('.csv')],
    'gtsrb': [f for f in os.listdir(folder_path) if 'gtsrb' in f and f.endswith('.csv')]    
}

def run_from_file_name(file_name):
    # returns "kd" from "report_fmnist_kd.csv"
    return file_name.split("_")[-1].split(".")[0]

def format_acc_value(v):
    return 'nan' if v == -1 else f'{v * 100:.2f}'

def format_acc(run, dataset):
    p = p_acc[dataset][run]
    c = c_acc[dataset][run]

    if p * c == max(best[dataset].values()):
        return f"\\textbf{{{format_acc_value(p)}}} & \\textbf{{{format_acc_value(c)}}}"
    else:
        return f"{format_acc_value(p)} & {format_acc_value(c)}"

def format_dl_weight(key, dataset):
    return 'nan' if dl_weight[dataset][key] == -1 else dl_weight[dataset][key]

def format_time(key, dataset):
    value = time[dataset][key]
    formatted = f"{value:.1f}"
    return 'nan' if value == -1 else f"\\qty{{{formatted}}}{{\\second}}"

p_acc = {}
c_acc = {}
dl_weight = {}
time = {}
best = {}

for dataset, reports in experiments.items():
    p_acc[dataset] = { 'baseline': -1, 'dl2': -1, 'g': -1, 'kd': -1, 'lk': -1, 'gg': -1, 'rc': -1, 'rc-phi': -1, 'rc-s': -1, 'yg': -1 }
    c_acc[dataset] = { 'baseline': -1, 'dl2': -1, 'g': -1, 'kd': -1, 'lk': -1, 'gg': -1, 'rc': -1, 'rc-phi': -1, 'rc-s': -1, 'yg': -1 }
    dl_weight[dataset] = { 'baseline': -1, 'dl2': -1, 'g': -1, 'kd': -1, 'lk': -1, 'gg': -1, 'rc': -1, 'rc-phi': -1, 'rc-s': -1, 'yg': -1 }
    time[dataset] = { 'baseline': -1, 'dl2': -1, 'g': -1, 'kd': -1, 'lk': -1, 'gg': -1, 'rc': -1, 'rc-phi': -1, 'rc-s': -1, 'yg': -1 }
    best[dataset] = { 'baseline': -1, 'dl2': -1, 'g': -1, 'kd': -1, 'lk': -1, 'gg': -1, 'rc': -1, 'rc-phi': -1, 'rc-s': -1, 'yg': -1 }

    for report in reports:
        file = os.path.join(folder_path, report)
        run = run_from_file_name(file)

        with open(file) as f:
            line = f.readline().strip()
            match = re.search(r'--dl-weight=([\d.]+)', line)
            assert match, "Value not found."

            dl_weight[dataset][run] = float(match.group(1))

        df = pd.read_csv(file, comment='#')

        p = df['Test-P-Acc'].values[-(len(df) // 10):]
        c = df['Test-C-Acc'].values[-(len(df) // 10):]
        s = p * c
        i = np.argmax(s)

        p_acc[dataset][run] = p[i]
        c_acc[dataset][run] = c[i]
        best[dataset][run] = p[i] * c[i]
        time[dataset][run] = df['Time'].values[1:].mean() # ignore the first row because it is epoch 0 (no training yet)

latex_fmnist_cifar10 = [
    '\\begin{tabular}{@{}lrrrrrrr@{}}',
    '\\toprule',
    '  & \\multicolumn{3}{c}{\\textbf{Fashion-MNIST}} & \\multicolumn{3}{c}{\\textbf{CIFAR-10}} \\\\ \\cmidrule(lr){2-4} \\cmidrule(lr){5-7}',
    '                                    & \\multicolumn{1}{c}{P}               & \\multicolumn{1}{c}{C}               & \\multicolumn{1}{c}{$\\lambda$}          & \\multicolumn{1}{c}{P}                & \\multicolumn{1}{c}{C}                & \\multicolumn{1}{c}{$\\lambda$}           \\\\ \\midrule',
    f'  Baseline                         & {format_acc("baseline", "fmnist")} & \\multicolumn{{1}}{{c}}{{--}}          & {format_acc("baseline", "cifar10")} & \\multicolumn{{1}}{{c}}{{--}}           \\\\',
    f'  DL2                              & {format_acc("dl2", "fmnist")}      & {format_dl_weight("dl2", "fmnist")}    & {format_acc("dl2", "cifar10")}      & {format_dl_weight("dl2", "cifar10")}    \\\\',
    f'  $I_\\text{{G}}$                  & {format_acc("g", "fmnist")}        & {format_dl_weight("g", "fmnist")}      & {format_acc("g", "cifar10")}        & {format_dl_weight("g", "cifar10")}      \\\\',
    f'  $I_\\text{{KD}}$                 & {format_acc("kd", "fmnist")}       & {format_dl_weight("kd", "fmnist")}     & {format_acc("kd", "cifar10")}       & {format_dl_weight("kd", "cifar10")}     \\\\',
    f'  $I_\\text{{\\L K}}$              & {format_acc("lk", "fmnist")}       & {format_dl_weight("lk", "fmnist")}     & {format_acc("lk", "cifar10")}       & {format_dl_weight("lk", "cifar10")}     \\\\',
    f'  $I_\\text{{GG}}$                 & {format_acc("gg", "fmnist")}       & {format_dl_weight("gg", "fmnist")}     & {format_acc("gg", "cifar10")}       & {format_dl_weight("gg", "cifar10")}     \\\\',
    f'  $I_\\text{{RC}}$                 & {format_acc("rc", "fmnist")}       & {format_dl_weight("rc", "fmnist")}     & {format_acc("rc", "cifar10")}       & {format_dl_weight("rc", "cifar10")}     \\\\',
    f'  $(I_\\text{{RC}})_{{s=9}}$       & {format_acc("rc-s", "fmnist")}     & {format_dl_weight("rc-s", "fmnist")}   & {format_acc("rc-s", "cifar10")}     & {format_dl_weight("rc-s", "cifar10")}   \\\\',
    f'  $(I_\\text{{RC}})_{{\\phi=x^2}}$ & {format_acc("rc-phi", "fmnist")}   & {format_dl_weight("rc-phi", "fmnist")} & {format_acc("rc-phi", "cifar10")}   & {format_dl_weight("rc-phi", "cifar10")} \\\\',
    f'  $I_\\text{{YG}}$                 & {format_acc("yg", "fmnist")}       & {format_dl_weight("yg", "fmnist")}     & {format_acc("yg", "cifar10")}       & {format_dl_weight("yg", "cifar10")}     \\\\ \\bottomrule',
    '\\end{tabular}'
]

latex_gtsrb = [
    '\\begin{tabular}{@{}lrrr@{}}',
    '  \\toprule',
    '  & \\multicolumn{3}{c}{\\textbf{GTSRB}} \\\\ \\cmidrule(lr){2-4}',
    '                       & \\multicolumn{1}{c}{P}              & \\multicolumn{1}{c}{C}              & \\multicolumn{1}{c}{$\\lambda$}      \\\\ \\midrule',
    f'  Baseline            & {format_acc("baseline", "gtsrb")} & \\multicolumn{{1}}{{c}}{{--}}      \\\\',
    f'  DL2                 & {format_acc("dl2", "gtsrb")}      & {format_dl_weight("dl2", "gtsrb")} \\\\',
    f'  $T_\\text{{G}}$     & {format_acc("g", "gtsrb")}        & {format_dl_weight("g", "gtsrb")}   \\\\',
    f'  $T_\\text{{\\L K}}$ & {format_acc("lk", "gtsrb")}       & {format_dl_weight("lk", "gtsrb")}  \\\\',
    f'  $T_\\text{{RC}}$    & {format_acc("rc", "gtsrb")}       & {format_dl_weight("rc", "gtsrb")}  \\\\',
    f'  $T_\\text{{YG}}$    & {format_acc("yg", "gtsrb")}       & {format_dl_weight("yg", "gtsrb")}  \\\\ \\bottomrule',
    '\\end{tabular}'
]

latex_fmnist_cifar10_times = [
    '\\begin{tabular}{@{}lrr@{}}',
    '  \\toprule',
    '                                    & \\textbf{Fashion-MNIST}                       & \\textbf{CIFAR-10}                             \\\\ \\midrule',
    f'  Baseline                         & {format_time("baseline", "fmnist")} & {format_time("baseline", "cifar10")} \\\\',
    f'  DL2                              & {format_time("dl2", "fmnist")}      & {format_time("dl2", "cifar10")}      \\\\',
    f'  $I_\\text{{G}}$                  & {format_time("g", "fmnist")}        & {format_time("g", "cifar10")}        \\\\',
    f'  $I_\\text{{KD}}$                 & {format_time("kd", "fmnist")}       & {format_time("kd", "cifar10")}       \\\\',
    f'  $I_\\text{{\\L K}}$              & {format_time("lk", "fmnist")}       & {format_time("lk", "cifar10")}       \\\\',
    f'  $I_\\text{{GG}}$                 & {format_time("gg", "fmnist")}       & {format_time("gg", "cifar10")}       \\\\',
    f'  $I_\\text{{RC}}$                 & {format_time("rc", "fmnist")}       & {format_time("rc", "cifar10")}       \\\\',
    f'  $(I_\\text{{RC}})_{{s=9}}$       & {format_time("rc-s", "fmnist")}     & {format_time("rc-s", "cifar10")}     \\\\',
    f'  $(I_\\text{{RC}})_{{\\phi=x^2}}$ & {format_time("rc-phi", "fmnist")}   & {format_time("rc-phi", "cifar10")}   \\\\',
    f'  $I_\\text{{YG}}$                 & {format_time("yg", "fmnist")}       & {format_time("yg", "cifar10")}       \\\\ \\bottomrule',
    '\\end{tabular}'
]

latex_gtsrb_times = [
    '\\begin{tabular}{@{}lr@{}}',
    '  \\toprule',
    '                       & \\textbf{GTSRB}                              \\\\ \\midrule',
    f'  Baseline            & {format_time("baseline", "gtsrb")} \\\\',
    f'  DL2                 & {format_time("dl2", "gtsrb")}      \\\\',
    f'  $T_\\text{{G}}$     & {format_time("g", "gtsrb")}        \\\\',
    f'  $T_\\text{{\\L K}}$ & {format_time("lk", "gtsrb")}       \\\\',
    f'  $T_\\text{{RC}}$    & {format_time("rc", "gtsrb")}       \\\\',
    f'  $T_\\text{{YG}}$    & {format_time("yg", "gtsrb")}       \\\\ \\bottomrule',
    '\\end{tabular}'
]

with open('table_fmnist_cifar10.tex', 'w') as file:
    file.write('\n'.join(latex_fmnist_cifar10))

with open('table_fmnist_cifar10_times.tex', 'w') as file:
    file.write('\n'.join(latex_fmnist_cifar10_times))

with open('table_gtsrb.tex', 'w') as file:
    file.write('\n'.join(latex_gtsrb))

with open('table_gtsrb_times.tex', 'w') as file:
    file.write('\n'.join(latex_gtsrb_times))