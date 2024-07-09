import sys 
import h5py
import numpy as np
# 讀取 MATLAB v7.3 文件
ABP_Raw = []
ABP_SPeaks = []
ABP_Turns = []
ABP_F = []

path = 'D:\\PulseDB\\PulseDB_Vital'

with h5py.File(sys.argv[1], 'r') as f:
    matdata = f['Subj_Wins']
    input(f'matdata.keys(): {matdata.keys()}')
    items = zip(matdata['ABP_Raw'][0], matdata['ABP_SPeaks'][0], matdata['ABP_Turns'][0], matdata['ABP_F'][0])
    for _ABP_Raw, _ABP_SPeaks,_ABP_Turns, _ABP_F in items:
        # input(f'{f[_ABP_Raw]}, {f[_ABP_SPeaks]}, {f[_ABP_F]}')
        ABP_Raw.append(f[_ABP_Raw][0])
        ABP_SPeaks.append(f[_ABP_SPeaks][0])
        ABP_Turns.append(f[_ABP_Turns][0])
        ABP_F.append(f[_ABP_F][0])

# ABP_Raw = np.array(ABP_Raw).flatten()
print(f'ABP_SPeaks: {ABP_SPeaks}')
# input(f'len(ABP_SPeaks): {len(ABP_SPeaks)}')
# print(f'ABP_Raw: {ABP_Raw}')
# print(f'len(ABP_Raw): {len(ABP_Raw)}')

for r in ABP_Turns:
    if 0. in r or r[-1] >= 1250:
        input(f'{r}, LENGTH: {len(r)}')
        break
    # input(f'{r}, LENGTH: {len(r)}')
# ABP_SPeaks = np.array(ABP_SPeaks[array_index]).flatten().astype(np.int64) - 1 - 2  # convert to 0-index and remove 4th order filter latency