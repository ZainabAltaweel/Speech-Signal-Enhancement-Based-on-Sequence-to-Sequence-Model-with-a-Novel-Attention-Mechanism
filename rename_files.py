import os
i = 0
files_path = "D:/UB_MS/Thesis/Seprating speech signals/DataSet/one_noise_dataset/noisy_part4/"
for filename in os.listdir(files_path): 
    dst ="noisy" + str(i) + ".wav"
    src = files_path + filename 
    dst = files_path + dst  
    # rename() function will 
    # # rename all the files 
    os.rename(src, dst) 
    i += 1
