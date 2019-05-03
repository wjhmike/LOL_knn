from django.shortcuts import render
from django.http import HttpResponse
from django.core.files.storage import FileSystemStorage
from django.contrib import messages
import os
import pandas as pd
from .knn import *
# Create your views here.

def home(request):
    context = {}
    global kd_tree
    global data
    if request.method == 'POST':
        if len(request.FILES) != 0:
            uploaded_file = request.FILES['dataset']
            data = pd.read_csv(uploaded_file, header=0, delimiter=',')
            column = {'t1_champ1id','t1_champ2id','t1_champ3id','t1_champ4id','t1_champ5id','t2_champ1id',
                      't2_champ2id','t2_champ3id','t2_champ4id','t2_champ5id','firstTower','firstInhibitor','firstBaron',
                      'firstDragon','firstRiftHerald','t1_towerKills','t1_inhibitorKills', 't1_baronKills',
                      't1_dragonKills','t2_towerKills','t2_inhibitorKills', 't2_baronKills','t2_dragonKills'}
            if column.issubset(data.columns) and len(data) > 10000:
                fs = FileSystemStorage()
                if fs.exists('dataset.csv'):
                    os.remove('data/dataset.csv')
                dataset_name = fs.save('dataset.csv', uploaded_file)
                dataset_path = fs.url(dataset_name)
                context['dataset_path'] = dataset_path
                messages.success(request, 'Dataset Uploaded Success')
                kd_tree = build_tree(data)
            else:
                messages.warning(request,'The format of dataset is not correct! Please resubmit!')
        elif request.POST.get('inlineRadioOptions') == 'beforeGame':
            if len(data)>0:
                sample = request.POST.dict()
                sample['t1_champ1id'] = int(sample['t1_champ1id'])
                sample['t1_champ2id'] = int(sample['t1_champ2id'])
                sample['t1_champ3id'] = int(sample['t1_champ3id'])
                sample['t1_champ4id'] = int(sample['t1_champ4id'])
                sample['t1_champ5id'] = int(sample['t1_champ5id'])
                sample['t2_champ1id'] = int(sample['t2_champ1id'])
                sample['t2_champ2id'] = int(sample['t2_champ2id'])
                sample['t2_champ3id'] = int(sample['t2_champ3id'])
                sample['t2_champ4id'] = int(sample['t2_champ4id'])
                sample['t2_champ5id'] = int(sample['t2_champ5id'])
                nearest_neigbors = knn_champ(7, sample, data)
                result1, result2 = predict(nearest_neigbors)
                context = {'team1': str(result1), 'team2': str(result2)}
            else:
                messages.warning(request, 'No dataset was uploaded!')
        elif request.POST.get('inlineRadioOptions') != 'inGame':
            if len(data)>0:
                sample = request.POST.dict()
                sample['t1_champ1id'] = int(sample['t1_champ1id'])
                sample['t1_champ2id'] = int(sample['t1_champ2id'])
                sample['t1_champ3id'] = int(sample['t1_champ3id'])
                sample['t1_champ4id'] = int(sample['t1_champ4id'])
                sample['t1_champ5id'] = int(sample['t1_champ5id'])
                sample['t2_champ1id'] = int(sample['t2_champ1id'])
                sample['t2_champ2id'] = int(sample['t2_champ2id'])
                sample['t2_champ3id'] = int(sample['t2_champ3id'])
                sample['t2_champ4id'] = int(sample['t2_champ4id'])
                sample['t2_champ5id'] = int(sample['t2_champ5id'])
                sample['firstTower'] = int(sample['firstTower'])
                sample['firstInhibitor'] = int(sample['firstInhibitor'])
                sample['firstBaron'] = int(sample['firstBaron'])
                sample['firstDragon'] = int(sample['firstDragon'])
                sample['firstRiftHerald'] = int(sample['firstRiftHerald'])
                sample['t1_towerKills'] = int(sample['t1_towerKills'])
                sample['t1_inhibitorKills'] = int(sample['t1_inhibitorKills'])
                sample['t1_baronKills'] = int(sample['t1_baronKills'])
                sample['t1_dragonKills'] = int(sample['t1_dragonKills'])
                sample['t2_towerKills'] = int(sample['t2_towerKills'])
                sample['t2_inhibitorKills'] = int(sample['t2_inhibitorKills'])
                sample['t2_baronKills'] = int(sample['t2_baronKills'])
                sample['t2_dragonKills'] = int(sample['t2_dragonKills'])
                key = get_key(sample)
                nearest_neigbors = knn(7, sample, kd_tree[key])
                result1, result2 = predict(nearest_neigbors)
                context = {'team1': str(result1), 'team2': str(result2)}
            else:
                messages.warning(request, 'No dataset was uploaded!')
    return render(request, 'LOL_Prediction/home.html', context)

def prediction(request):
    # Handle file upload
    if request.method == 'POST' and request.FILES['dataset']:
        myfile = request.FILES['dataset']
        fs = FileSystemStorage()
        filename = fs.save(myfile.name, myfile)
        uploaded_file_url = fs.url(filename)
        return render(request, 'LOL_Prediction/prediction.html')
    else:
        return render(request, 'LOL_Prediction/home.html')
