# Copy Train_Test_Split INCLUDE folder to ./ to run this code


from math import ceil
import random
import os
import csv
from meta_generator import generate_meta

val_split = 0.1

if __name__ == "__main__":
    root = 'C:\\Sreelakshmi V\\sl-hwgat-main'
    data_path = os.path.join(root, "INCLUDE")
    split_path = os.path.join(root, "Train_Test_Split")

    print(data_path, split_path, sep="\n")

    # initializing the titles and rows list
    header = []
    rows = []
    vocab = []
    
    train_rows = {}
    # reading csv file
    with open(split_path + '/train_include.csv', 'r') as csvfile:
        print("inside train_include.csv")
        # creating a csv reader object
        csvreader = csv.reader(csvfile)
        
        # extracting field names through first row
        header = next(csvreader)
    
        # extracting each data row one by one
        for row in csvreader:
            vid_path = os.path.normpath(os.path.join(root, row[3]))
            print(vid_path)
            try:
                f = open(vid_path, 'r')
                f.close()

                vid_parts = os.path.normpath(vid_path).split(os.sep)  # Normalize path for Windows & Linux
                try:
                    if 'Extra' in vid_path or 'extra' in vid_path:
                        category_part = vid_parts[-3]  # Go 3 levels up
                    else:
                        category_part = vid_parts[-2]  # Go 2 levels up

                    # Extract category name safely (if it has a dot, take the second part)
                    cls = category_part.split('.')[-1].strip().lower()
                except IndexError:
                    print(f"Error extracting class for: {vid_path}")
                    cls = "unknown"  # Assign a default category if extraction fails
                print("Working boiiiiiii")
                if cls not in vocab:
                    vocab.append(cls)

                if train_rows.get(cls) == None:
                    train_rows[cls] = []
                train_rows[cls].append([os.path.join('INCLUDE', row[3]), vid_path.split('/')[-1], cls, 'train'])
            except Exception as e:
                print(vid_path, e)
                continue
    
    vocab.sort()

    for cls in train_rows.keys():
        idxs = random.sample(range(len(train_rows[cls])), ceil(len(train_rows[cls])*val_split))
        for idx in idxs:
            train_rows[cls][idx][3] = 'val'
    
    id = 0
    for cls in train_rows.keys():
        for row in train_rows[cls]:
            rows.append(["{:07d}".format(id), row[0], row[1], row[2], row[3]])
            id += 1

    test_rows = []

    # reading csv file
    with open(split_path + '/test_include.csv', 'r') as csvfile:
        # creating a csv reader object
        csvreader = csv.reader(csvfile)
        
        # extracting field names through first row
        header = next(csvreader)
    
        # extracting each data row one by one
        for row in csvreader:
            vid_path = os.path.normpath(os.path.join(root, row[3]))
            try:
                f = open(vid_path, 'r')
                f.close()

                vid_parts = os.path.normpath(vid_path).split(os.sep)  # Normalize path for Windows & Linux
                try:
                    if 'Extra' in vid_path or 'extra' in vid_path:
                        category_part = vid_parts[-3]  # Go 3 levels up
                    else:
                        category_part = vid_parts[-2]  # Go 2 levels up

                    # Extract category name safely (if it has a dot, take the second part)
                    cls = category_part.split('.')[-1].strip().lower()
                    print("Working Boiii 2222!!!!")
                except IndexError:
                    print(f"Error extracting class for: {vid_path}")
                    cls = "unknown"  # Assign a default category if extraction fails

                test_rows.append([vid_path, cls])
                rows.append(["{:07d}".format(id), os.path.join('INCLUDE', row[3]), vid_path.split('/')[-1], cls, 'test'])
                id += 1
            except:
                continue
    
    print(f'Data Path : {data_path},\nRows: {rows},\n\n\nVovabs: {vocab}')

    generate_meta(data_path, rows, vocab)