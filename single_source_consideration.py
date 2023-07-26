import pandas as pd
import os
import glob

src_path = '/home/balaji/Documents/AED_research/seld-dcase2022-main/orginal_csv_files/dev-test-sony/'
dst_path = '/home/balaji/Documents/AED_research/seld-dcase2022-main/data_set_mono/metadata_dev/dev-test-sony/'
all_csv_files = glob.glob(src_path+ '*.csv')
# print(all_csv_files)
for file in all_csv_files:
    print(file)
    file_name = os.path.split(file)


    data = pd.read_csv(file)

    rslt_df = data[data.iloc[:, 2] == 0]

    # rslt_df.reset_index(drop=True)
    # print(rslt_df)
    rslt_df.to_csv(str(dst_path+file_name[1]))
# for i in range(len(source_info)):
#     if source_info[i]== 0:
#         filtered_data =data.iloc[i,:]
#         print(filtered_data)



