import numpy as np

data_np= np.load("/home/balaji/Documents/AED_research/seld-dcase2022-main/data_set_mono/feat_label/mic_dev_label/fold3_room4_mix001.npy")
print(data_np)
with open('data.txt', 'wb') as f:
    np.save(f, data_np, allow_pickle=False)