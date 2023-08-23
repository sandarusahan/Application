import os
import glob
import random

dir_fake = 'D:\MSc\OneDrive - MMU\Documents\MSc\Dissertation\Application\Output\Fake1'
dir_real = 'D:\MSc\OneDrive - MMU\Documents\MSc\Dissertation\Application\Output\Real1'
files_fake = glob.glob(os.path.join(dir_fake, '*.mp4'))
files_real = glob.glob(os.path.join(dir_real, '*.mp4'))


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


def create_csv(files, csv_filename):
    with open(csv_filename, 'w') as f:
        for file in files:
            is_real = 1 if 'Real' in file else 0
            f.write(file+","+str(is_real) + '\n')


if __name__ == "__main__":
    train_fake, val_fake, test_fake = random_split_array(files_fake)
    train_real, val_real, test_real = random_split_array(files_real)
    train = train_fake + train_real
    val = val_fake + val_real
    test = test_fake + test_real

    create_csv(train, 'train2.csv')
    create_csv(val, 'val2.csv')
    create_csv(test, 'test2.csv')

    print(len(train))
    print(len(val))
    print(len(test))

    print("Done")
