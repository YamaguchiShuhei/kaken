import collections
import random
import sys
import os

def read_words(filename):
    f = open(filename, "r")
    line = f.readline()
    list = []
    while line:
        list.append(line.split())
        line = f.readline()
    word_count = 0
    sent_count = 0
    word_list = []
    for line in list:
        if line[0] == "#":
            sent_count += 1
        elif len(line) > 6:
            word_count += 1
            word_list.append(line[0])
    return sent_count, word_count, word_list

def character_id(word_list):
    cha = []
    for word in word_list:
        cha.extend(word)
    counter = collections.Counter(cha)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
    words, _ = list(zip(*count_pairs))
    cha_id = dict(zip(words, range(len(words))))
    return cha_id
    
if __name__ == "__main__":
    data_path = sys.argv[1]
    sent_count, word_count, word_list = read_words(data_path)
    print("sentence " + str(sent_count))
    print("word " + str(word_count))
    cha_id = character_id(word_list)
    print("cha_count " + str(len(cha_id)))
