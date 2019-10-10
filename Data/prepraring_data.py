# -*- coding: utf-8 -*-

import os
import pandas as pd
from Preprocessing import Preprocessing
from pyvi import ViTokenizer

pre = Preprocessing()


def make_new_list_line(list_line, filename):
    train_file = open(str(filename), 'w', encoding='utf-8')
    new_list_line = []
    for i, line in enumerate(list_line):
        # print(str(i), len(list_line))
        temp = []
        t = 1
        for token in line.split():
            new_token = pre.sylabelize(token)
            # new_token = pre.get_nomal_sentence(new_token)
            if new_token.isalpha():
                temp.append(new_token)
            elif new_token.isnumeric():
                temp.append('@NUM@')
            else:
                t = 0
        if len(temp) < 5:
            continue
        if t == 1:
            new_list_line.append(' '.join(temp))
            print(temp)
    string_lines = '\n'.join(new_list_line)
    train_file.write(string_lines)
    train_file.close()


def main():
    tranform_folder_train_to_lines('Data')


def tranform_folder_train_to_lines(folder_path, des_path='/'):
    # Getting the current work directory (cwd)
    if os.path.isdir(folder_path):
        thisdir = folder_path
    else:
        thisdir = os.getcwd() + '/' + folder_path
    list_lines = []
    for r, d, f in os.walk(thisdir):
        for file in f:
            if ".txt" in file:
                dir_file = os.path.join(r, file)
                lines = open(dir_file, 'r', encoding='utf_16_le').readlines()
                # lines = open(dir_file, 'r', encoding='utf-8').readlines()
                for line in lines:
                    x = line[-1]
                    line = line.replace(x, '').lower()
                    print(len(line.split()), line)
                    if len(line.strip().split()) > 4:
                        # new_line = ViTokenizer.tokenize(line)
                        list_lines.extend(line.strip().split('.'))
    make_new_list_line(list_lines, des_path + 'TrainingData.txt')


if __name__ == '__main__':
    main()
