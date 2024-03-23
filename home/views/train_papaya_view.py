from django.shortcuts import render
import matplotlib.pyplot as plt
from configs.settings import BASE_DIR
import itertools
import numpy as np
import pandas as pd
import os
import cv2
import tensorflow as tf
from keras.layers import Dropout, Flatten, Dense, Conv2D, MaxPooling2D
from keras import Sequential
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from training_tool.models import TrainingTool

def plot_confusion_matrix(
        cm, 
        classes,
        normalize=False,
        title='Confusion matrix',
        cmap=plt.cm.Blues
    ):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True class')
    plt.xlabel('Predicted class')

def train_papaya_view(request):
    
    '''Docstring here.'''
    arquivo = None
    mensagens = []
    if request.method=='POST':
        # ETAPA 1 - Realizando importações 

        
        

        #Carregando DATASET
        folder = os.path.join(BASE_DIR, 'papaya_dataset_/papaya_dataset_01/')

        # ETAPA 2 - Carregando o dataset
        image_width = 64
        image_height = 64
        channels = 3

        train_files = []
        i=0
        for estado in ['mature', 'partiallymature', 'unmature']:
            onlyfiles = [f for f in os.listdir(folder + '/' + str(estado)) if os.path.isfile(os.path.join(folder + '/' + str(estado), f))]
            for _file in onlyfiles:
                train_files.append(_file)

        dataset = np.ndarray(shape=(len(train_files), image_height, image_width, channels),dtype=np.float32)
        y_dataset = []

        i = 0
        for estado in ['mature', 'partiallymature', 'unmature']:
            onlyfiles = [f for f in os.listdir(folder + '/' + str(estado)) if os.path.isfile(os.path.join(folder + '/' + str(estado), f))]
            for _file in onlyfiles:
                ###
                img = cv2.imread(os.path.join(folder, estado, _file))
                img_resized = cv2.resize(img, (image_width, image_height))

                dataset[i] = img_resized
                mapping = {'mature': 0, 'partiallymature': 1, 'unmature': 2}
                y_dataset.append(mapping[estado])
                ###
                i += 1
                if i % 250 == 0:
                    print("%d images to array" % i)
        print("All images to array!")

        # Normalizando os dados
        dataset = dataset.astype('float32')
        dataset /= 255

        pixels = np.array(dataset[0], dtype='float32')
        plt.imshow(cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB))
        plt.show()
        plt.close()

        
        n_classes = len(set(y_dataset))
        print(n_classes)

        y_dataset_ = to_categorical(y_dataset, n_classes)

        X_train, X_test, y_train, y_test = train_test_split(dataset, y_dataset_, test_size=0.2)
        print("Train set size: {0}, Test set size: {1}".format(len(X_train), len(X_test)))

        
        # ETAPA 3 - Criando o modelo

        datagen = ImageDataGenerator(rotation_range=90, shear_range=0.2, horizontal_flip=True, fill_mode='nearest')

        datagen.fit(X_train)

        model = Sequential()
        model.add(Conv2D(128, (3, 3), activation='relu', input_shape=(64, 64, 3)))
        model.add(MaxPooling2D((2, 2)))
        model.add(Flatten())

        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(3, activation='softmax'))
        
        model.summary()
        # Taxa de aprendizado
        opt = tf.keras.optimizers.Adam(learning_rate=0.001)

        # Compilando o modelo
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

        # Parada antecipada
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', mode = 'max', patience = 5)

        # Treinando o modelo
        history = model.fit(X_train, y_train, validation_split=0.2, epochs=20, callbacks = [early_stopping])
        

        # Gráfico de treinamento e validação da função perda
        plt.plot(history.history['loss'], label = 'loss')
        plt.plot(history.history['val_loss'], label = 'val_loss')
        plt.title("Função de perda")
        plt.xlabel('Épocas')
        plt.ylabel('MSE')
        plt.legend(["Treinando"], loc='upper left')
        #plt.grid(True)
        plt.savefig(os.path.join(BASE_DIR, 'graphs/graph_erro.png'))
        plt.close()
        

        # Gráfico de treinamento e validação da acurácia
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.yscale("log")
        plt.title('Acurácia')
        plt.ylabel('Acurácia')
        plt.xlabel('Épocas')
        plt.legend(['Treino', 'Validação'])
        #plt.grid(True)
        plt.savefig(os.path.join(BASE_DIR, 'graphs/graph_treinamento_validacao_acuracia.png'))
        plt.close()

        preds = model.predict(X_test)
        
        n = 10
        total_images = len(X_test)
        for t in range(total_images // n):
            plt.figure(figsize=(15,15))

            for i in range(n*t, min(n*(t+1), total_images)):
                plt.subplot(1, n, i + 1 - n*t)
                plt.imshow(cv2.cvtColor(X_test[i], cv2.COLOR_BGR2RGB), cmap='gray')
                plt.title('Label: {}\nPredicted: {}'.format(np.argmax(y_test[i]), np.argmax(preds[i])))
                plt.axis('off')
            plt.savefig(os.path.join(BASE_DIR, 'images_pred/graph_predicao_{}.png'.format(t)))
            plt.close()


        y_test_ = [np.argmax(x) for x in y_test]
        preds_ = [np.argmax(x) for x in preds]

        cm = confusion_matrix(y_test_, preds_)
        plot_confusion_matrix(cm, classes=['mature', 'partiallymature', 'unmature'], title='Confusion matrix')
        plt.savefig(os.path.join(BASE_DIR, 'graphs/graph_confusion_matrix.png'))
        plt.close()

            

        # Calcular acurácia
        accuracy = accuracy_score(y_test_, preds_)
        print("Acurácia:", accuracy)

        # Calcular precisão
        precision = precision_score(y_test_, preds_, average='macro')
        print("Precisão:", precision)

        # Calcular recall
        recall = recall_score(y_test_, preds_, average='macro')
        print("Recall:", recall)

        # Calcular F1 score
        f1 = f1_score(y_test_, preds_, average='macro')
        print("F1-score:", f1)

        # Convertendo o histórico de treinamento em um DataFrame do Pandas
        historico = pd.DataFrame(history.history)

        # Salvando o DataFrame em um arquivo CSV
        historico.to_csv(os.path.join(BASE_DIR, 'historico.csv'), index=False)

        # saving model
        model.save(os.path.join(BASE_DIR, 'models_train/modelfile.h5'))

        ### Futura Versão
        """ file_path = os.path.join(BASE_DIR, 'models_train/modelfile.h5')
        with open(file_path, 'rb') as file:
            f_modelo = file.read()

        file_path = os.path.join(BASE_DIR, 'graphs/graph_erro.png')
        with open(file_path, 'rb') as file:
            f_graph_erro = file.read()

        file_path = os.path.join(BASE_DIR, 'graphs/graph_treinamento_validacao_acuracia.png')
        with open(file_path, 'rb') as file:
            f_graph_acuracia = file.read()

        file_path = os.path.join(BASE_DIR, 'graphs/graph_confusion_matrix.png')
        with open(file_path, 'rb') as file:
            f_graph_cm = file.read()

        file_path = os.path.join(BASE_DIR, 'historico.csv')
        with open(file_path, 'rb') as file:
            f_historico = file.read()

        TrainingTool.objects.create(
            model=f_modelo,
            graph_erro=f_graph_erro,
            graph_acuracia=f_graph_acuracia,
            graph_confusion_matrix=f_graph_cm,
            historico=f_historico
        ) """

        mensagens.append(f"O modelo foi treinado com sucesso.")

        
    context = {
        "arquivo": arquivo,
        "mensagens": mensagens,
        
    }

    return render(
        request,
        'home/index_train_model.html',
        context,
    )