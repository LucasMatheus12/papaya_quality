from django.contrib.auth.decorators import login_required
from django.shortcuts import render
import numpy as np
import os
from configs.settings import BASE_DIR
import matplotlib
from PIL import Image
import matplotlib
matplotlib.use('agg')
import tensorflow as tf

def teste_treinado_view(request):
    '''Docstring here.'''
    arquivo = None
    mensagem = []
    if request.method=='POST':
        arquivo = request.FILES.get('arquivo')
        
        
        # Salva o arquivo temporariamente
        file_path = os.path.join(BASE_DIR, arquivo.name)
        with open(file_path, 'wb+') as destination:
            for chunk in arquivo.chunks():
                destination.write(chunk)

        # Carrega a imagem de teste
        test_image = Image.open(file_path)
        test_image = test_image.resize((227, 227))  # Redimensiona para o tamanho esperado pelo modelo
        test_image = tf.keras.preprocessing.image.img_to_array(test_image)
        test_image = test_image / 255.0  # Normaliza a imagem

        # Adiciona uma dimensão extra para a amostra (batch)
        test_image = tf.expand_dims(test_image, axis=0)

        # Carrega o modelo treinado
        model = tf.keras.models.load_model(os.path.join(BASE_DIR,'model_train.h5'))

        # Faz a previsão
        prediction = model.predict(test_image)

        # Determina a classe com maior probabilidade
        class_index = np.argmax(prediction)
        
        # Mapeia o índice da classe para a classe correspondente
        classes = ['maduro', 'parcialmente maduro', 'não maduro']
        resultado = classes[class_index]
        mensagem.append(f"O modelo prevê que a fruta é {resultado}.")
        # Exclui o arquivo temporário
        os.remove(file_path)
        
    context = {
        "arquivo": arquivo,
        "mensagem": mensagem
    }

    return render(
        request,
        'home/index.html',
        context,
    )
