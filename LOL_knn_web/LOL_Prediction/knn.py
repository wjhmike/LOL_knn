import numpy as np
import pandas as pd
import sys


def compute_distence(inputFeature, simples):
    last_distence = sys.maxsize
    nearest_index = 0
    for index, row in simples.iterrows():
        distence = 0
        champ_group1 = [row['t1_champ1id'], row['t1_champ2id'], row['t1_champ3id'], row['t1_champ4id'],
                        row['t1_champ5id']]
        champ_group2 = [row['t2_champ1id'], row['t2_champ2id'], row['t2_champ3id'], row['t2_champ4id'],
                        row['t2_champ5id']]
        if inputFeature['t1_champ1id'] not in champ_group1:
            distence += 1
        if inputFeature['t1_champ2id'] not in champ_group1:
            distence += 1
        if inputFeature['t1_champ3id'] not in champ_group1:
            distence += 1
        if inputFeature['t1_champ4id'] not in champ_group1:
            distence += 1
        if inputFeature['t1_champ5id'] not in champ_group1:
            distence += 1
        if inputFeature['t2_champ1id'] not in champ_group2:
            distence += 1
        if inputFeature['t2_champ2id'] not in champ_group2:
            distence += 1
        if inputFeature['t2_champ3id'] not in champ_group2:
            distence += 1
        if inputFeature['t2_champ4id'] not in champ_group2:
            distence += 1
        if inputFeature['t2_champ5id'] not in champ_group2:
            distence += 1
        if inputFeature['firstBlood'] != row['firstBlood']:
            distence += 1
        if inputFeature['firstTower'] != row['firstTower']:
            distence += 1
        if inputFeature['firstInhibitor'] != row['firstInhibitor']:
            distence += 1
        if inputFeature['firstBaron'] != row['firstBaron']:
            distence += 1
        if inputFeature['firstDragon'] != row['firstDragon']:
            distence += 1
        if inputFeature['firstRiftHerald'] != row['firstRiftHerald']:
            distence += 1
        distence += abs(row['t1_towerKills'] - inputFeature['t1_towerKills'])
        distence += abs(row['t1_inhibitorKills'] - inputFeature['t1_inhibitorKills'])
        distence += abs(row['t1_baronKills'] - inputFeature['t1_baronKills'])
        distence += abs(row['t1_dragonKills'] - inputFeature['t1_dragonKills'])
        distence += abs(row['t2_towerKills'] - inputFeature['t2_towerKills'])
        distence += abs(row['t2_inhibitorKills'] - inputFeature['t2_inhibitorKills'])
        distence += abs(row['t2_baronKills'] - inputFeature['t2_baronKills'])
        distence += abs(row['t2_dragonKills'] - inputFeature['t2_dragonKills'])

        if distence < last_distence:
            last_distence = distence
            nearest_index = index
    # print("distence: ", last_distence)
    return simples.loc[nearest_index]


def compute_distence_champ(inputFeature, simples):
    last_distence = sys.maxsize
    nearest_index = 0
    for index, row in simples.iterrows():
        distence = 0
        champ_group1 = [row['t1_champ1id'], row['t1_champ2id'], row['t1_champ3id'], row['t1_champ4id'],
                        row['t1_champ5id']]
        champ_group2 = [row['t2_champ1id'], row['t2_champ2id'], row['t2_champ3id'], row['t2_champ4id'],
                        row['t2_champ5id']]
        if inputFeature['t1_champ1id'] not in champ_group1:
            distence += 1
        if inputFeature['t1_champ2id'] not in champ_group1:
            distence += 1
        if inputFeature['t1_champ3id'] not in champ_group1:
            distence += 1
        if inputFeature['t1_champ4id'] not in champ_group1:
            distence += 1
        if inputFeature['t1_champ5id'] not in champ_group1:
            distence += 1
        if inputFeature['t2_champ1id'] not in champ_group2:
            distence += 1
        if inputFeature['t2_champ2id'] not in champ_group2:
            distence += 1
        if inputFeature['t2_champ3id'] not in champ_group2:
            distence += 1
        if inputFeature['t2_champ4id'] not in champ_group2:
            distence += 1
        if inputFeature['t2_champ5id'] not in champ_group2:
            distence += 1

        if distence < last_distence:
            last_distence = distence
            nearest_index = index
    # print("distence: ", last_distence)
    return simples.loc[nearest_index]

def knn(k, inputFeature, simples):
    nearest_neigbors = []
    simples_copy = simples.copy()
    for i in range(k):
        row = compute_distence(inputFeature, simples_copy)
        # print(row.name)
        simples_copy = simples_copy.drop(row.name)
        nearest_neigbors.append(row)
    return nearest_neigbors

def knn_champ(k, inputFeature, simples):
    nearest_neigbors = []
    simples_copy = simples.copy()
    for i in range(k):
        row = compute_distence_champ(inputFeature, simples_copy)
        #print(row.name)
        simples_copy = simples_copy.drop(row.name)
        nearest_neigbors.append(row)
    return nearest_neigbors


def predict(nearest_neigbors):
    t1_count = 0
    t2_count = 0
    for row in nearest_neigbors:
        if row['winner'] == 1:
            t1_count += 1
        else:
            t2_count = 1
    return t1_count/7, t2_count/7


def build_tree(train_x):
    kd_tree = {}
    key = ''
    for a in (1, 2):
        for b in (1, 2):
            for c in (1, 2):
                for d in (1, 2):
                    key = str(a) + str(b) + str(c) + str(d)
                    kd_tree[key] = pd.DataFrame()
                    key = ''

    for index, row in train_x.iterrows():
        key = get_key(row)
        kd_tree[key] = kd_tree[key].append(row)
    return kd_tree


def get_key(simple):
    key = ''
    if simple['t1_towerKills'] < 5:
        key = key + '1'
    else:
        key = key + '2'
    if simple['t1_baronKills'] < 1:
        key = key + '1'
    else:
        key = key + '2'
    if simple['t1_dragonKills'] < 3:
        key = key + '1'
    else:
        key = key + '2'
    if simple['t1_riftHeraldKills'] < 1:
        key = key + '1'
    else:
        key = key + '2'
    return key