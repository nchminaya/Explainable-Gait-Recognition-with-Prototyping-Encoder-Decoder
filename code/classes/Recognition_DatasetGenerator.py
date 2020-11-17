# csulb-datascience
#
# Authors: 
#      Sella Bae,     email: sella.bae@student.csulb.edu
#
# Class version: 1.0
# Date: Nov 2020
#
# Include a reference to this site if you will use this code.

import random
import os
from datetime import datetime
from Recognition_Dataset_V1 import Dataset

class DatasetGenerator:
    def __init__(self, dirPath, fileName):
        self.users = Dataset(dirPath, fileName).user
        self.dirPath = dirPath
        self.fileName = fileName
        self.datasets = []
    
    def gen_datasets(self, n, numberPeopleTraining, numberPeopleValidation, verbose=0):
      """Generate n number of datasettings with evenly distributed training sets of size numberPeopleTraining
      """
        print(f"Generate {n} datasettings with evenly distributed training sets")
        print(f" from '{self.fileName}'")
        id_combs = list(random_combinations(self.users, numberPeopleTraining, n))

        self.datasets = []
        
        for i, ids in enumerate(id_combs):
            d = Dataset(self.dirPath, self.fileName)
            d.split_by_id(ids, numberPeopleValidation)

            self.datasets.append(d)
            if verbose: print(f"{i+1} \ttraining ids: {sorted(list(d.trainingSet.keys()))}")

        return self.datasets
    
    def save(self, storePath, prefix, verbose=0):
      """Save datasets as .npy files into storePath with prefix
      """
        if self.datasets == []:
            print('no datasets to save. Generate datasets with gen_datasets() first.')
            
        print(f"\nSave {len(self.datasets)} datasettings into '{storePath}'")
        if not os.path.isdir(storePath):
            os.makedirs(storePath)

        for i,d in enumerate(self.datasets):
            fname = prefix + str(i+1) + ".npy"
            d.saveSets(storePath, fname)
            if verbose: print(fname)
        
        with open(os.path.join(storePath,'readme.md'), 'w') as f:
            f.write(f"These datasets were generated from '{self.fileName}'\n")
            f.write(str(datetime.now()) + '\n')
        

def random_combinations(iterable, csize, n):
    """Generate approximately evenly distributed n combinations of csize with distinct random items in iterable
    """
    iter_copy = list(iterable)

    if csize > len(iter_copy):
        raise ValueError("combination size 'csize' can't exceed the length of 'iterable'")

    for i in range(n):
        comb = []
        for j in range(csize):
            if not iter_copy:
                iter_copy = list(iterable)

            randi = random.randint(0, len(iter_copy) - 1)

            # if iter_copy was reinstantiated, comb can have duplicate items in it
            while iter_copy[randi] in comb:
                randi = random.randint(0, len(iter_copy) - 1)
            comb.append(iter_copy.pop(randi))

        yield comb    # yield n combinations in total

def count_combinations(combinations):
    counts = {k: 0 for k in combinations}
    for comb in combinations:
        for e in comb:
              counts[e] += 1
    return counts
