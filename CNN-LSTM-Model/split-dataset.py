import os
import glob
import random
import json
import shutil

dir_fake = 'D:\MSc\OneDrive - MMU\Documents\MSc\Dissertation\Application\Output\Fake-FF'
dir_real = 'D:\MSc\OneDrive - MMU\Documents\MSc\Dissertation\Application\Output\Real-FF'
dir_fake_fr = 'D:\MSc\OneDrive - MMU\Documents\MSc\Dissertation\Application\Output\Fake-FR'
dir_real_fr = 'D:\MSc\OneDrive - MMU\Documents\MSc\Dissertation\Application\Output\Real-FR'

dfdc_ds = 'D:\MSc\OneDrive - MMU\Documents\MSc\Dissertation\Datasets\\train_sample_videos'
files_fake = glob.glob(os.path.join(dir_fake, '*.mp4'))
files_fake_fr = glob.glob(os.path.join(dir_fake_fr, '*.mp4'))
files_real = glob.glob(os.path.join(dir_real, '*.mp4'))
files_real_fr = glob.glob(os.path.join(dir_real_fr, '*.mp4'))



def random_split_array(files, train_ratio=0.6, val_ratio=0.20, test_ratio=0.20):
    # Calculate the sizes of the three parts
    train_size = int(len(files) * train_ratio)
    val_size = int(len(files) * val_ratio)
    test_size = int(len(files) * test_ratio)

    # Divide the original list into three parts
    train = random.sample(files, train_size)
    remaining_files = [file for file in files if file not in train]
    val = random.sample(remaining_files, val_size)
    test = [file for file in remaining_files if file not in val]

    return train, val, test

def read_metadata_copy_files(path):
    originals = []
    with open(path, 'r') as f:
        json_data = json.load(f)
        vid_list = json_data.keys()
        for ori in vid_list:
            if json_data[ori]['label'] == 'REAL':
                originals.append(ori)
        for vid in vid_list:
            if json_data[vid]['label'] == 'FAKE':
                if json_data[vid]["original"] in originals:
                    originals.remove(json_data[vid]["original"])
                    shutil.copy2(os.path.join(dfdc_ds, vid), 'D:\MSc\OneDrive - MMU\Documents\MSc\Dissertation\Datasets\\train_sample_videos\\fake')
            else:
                shutil.copy2(os.path.join(dfdc_ds, vid), 'D:\MSc\OneDrive - MMU\Documents\MSc\Dissertation\Datasets\\train_sample_videos\\real') 
    

def create_csv(files, csv_filename):
    with open(csv_filename, 'w') as f:
        for file in files:
            is_real = 1 if 'Real' in file else 0
            f.write(file+","+str(is_real) + '\n')


if __name__ == "__main__":
    train_fake, val_fake, test_fake = random_split_array(files_fake)
    train_fake_fr, val_fake_fr, test_fake_fr = random_split_array(files_fake_fr)
    train_real, val_real, test_real = random_split_array(files_real)
    train_real_fr, val_real_fr, test_real_fr = random_split_array(files_real_fr)

    train = train_fake + train_real + train_fake_fr + train_real_fr
    val = val_fake + val_real + val_fake_fr + val_real_fr
    test = test_fake + test_real + test_fake_fr + test_real_fr

    create_csv(train, 'train3.csv')
    create_csv(val, 'val3.csv')
    create_csv(test, 'test3.csv')

    print(len(train))
    print(len(val))
    print(len(test))
    # read_metadata_copy_files('D:\MSc\OneDrive - MMU\Documents\MSc\Dissertation\Datasets\\train_sample_videos\metadata.json')
    print("Done")
