from django.contrib.auth.decorators import login_required
from django.shortcuts import render
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
from configs.settings import BASE_DIR



def home_view(request):
    '''Docstring here.'''
    arquivo = None
    mensagem = []
    mature_dir = os.path.join(BASE_DIR, 'papaya_dataset_/papaya_dataset_01/mature/')
    partiallymature_dir = os.path.join(BASE_DIR, 'papaya_dataset_/papaya_dataset_01/partiallymature/')
    unmature_dir = os.path.join(BASE_DIR, 'papaya_dataset_/papaya_dataset_01/unmature/')
    if request.method=='POST':
        arquivo = request.FILES.get('arquivo')
        for dirname, _, filenames in os.walk(os.path.join(BASE_DIR, 'papaya_dataset_/')):
            for filename in filenames:
                mensagem.append(os.path.join(dirname, filename))
        

        mature_images = os.listdir(mature_dir)
        partiallymature_images = os.listdir(partiallymature_dir)
        unmature_images = os.listdir(unmature_dir)
        mensagem.append(len(mature_images))
        mensagem.append(len(partiallymature_images))
        mensagem.append(len(unmature_images))
        train_mature_images = mature_images[:int(.8*(len(mature_images)))]
        val_mature_images = mature_images[int(.8*(len(mature_images))):]

        train_partiallymature_images = partiallymature_images[:int(.8*(len(partiallymature_images)))]
        val_partiallymature_images = partiallymature_images[int(.8*(len(partiallymature_images))):]

        train_unmature_images = unmature_images[:int(.8*(len(unmature_images)))]
        val_unmature_images = unmature_images[int(.8*(len(unmature_images))):]
        # Gerando dados 
        train_dir = './train_data/'
        val_dir = './val_data/'
        os.makedirs(os.path.join(BASE_DIR, train_dir, 'mature/'))
        os.makedirs(os.path.join(BASE_DIR, train_dir, 'partiallymature/'))
        os.makedirs(os.path.join(BASE_DIR, train_dir, 'unmature/'))

        os.makedirs(os.path.join(BASE_DIR, val_dir, 'mature/'))
        os.makedirs(os.path.join(BASE_DIR, val_dir, 'partiallymature/'))
        os.makedirs(os.path.join(BASE_DIR, val_dir, 'unmature/'))


        for image in train_mature_images:
            src = mature_dir + image
            dst = train_dir + 'mature/'
            shutil.copy(src, dst)

        for image in train_partiallymature_images:
            src = partiallymature_dir + image
            dst = train_dir + 'partiallymature/'
            shutil.copy(src, dst)

        for image in train_unmature_images:
            src = unmature_dir + image
            dst = train_dir + 'unmature/'
            shutil.copy(src, dst)

        for image in val_mature_images:
            src = mature_dir + image
            dst = val_dir + 'mature/'
            shutil.copy(src, dst)

        for image in val_partiallymature_images:
            src = partiallymature_dir + image
            dst = val_dir + 'partiallymature/'
            shutil.copy(src, dst)

        for image in val_unmature_images:
            src = unmature_dir + image
            dst = val_dir + 'unmature/'
            shutil.copy(src, dst)

    context = {
        "arquivo": arquivo,
        "mensagem": mensagem
    }

    return render(
        request,
        'home/index.html',
        context,
    )
