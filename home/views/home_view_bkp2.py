from django.shortcuts import render
import matplotlib.pyplot as plt
import os
import shutil
from configs.settings import BASE_DIR
import matplotlib.image as mpimg
import random
import matplotlib
import matplotlib
matplotlib.use('agg')
import tensorflow as tf
import random

def view_random_image(target_dir, target_class):
    target_folder = target_dir + 'mature' + '/'
    random_image = random.choice(os.listdir(target_folder))
    
    img = mpimg.imread(target_folder + random_image)
    print(img.shape)
    plt.title('mature')
    plt.imshow(img)
    plt.axis('off')

def home_view(request):
    '''Docstring here.'''
    arquivo = None
    mensagem = []
    mature_dir = os.path.join(BASE_DIR, 'papaya_dataset_/papaya_dataset_01/mature/')
    partiallymature_dir = os.path.join(BASE_DIR, 'papaya_dataset_/papaya_dataset_01/partiallymature/')
    unmature_dir = os.path.join(BASE_DIR, 'papaya_dataset_/papaya_dataset_01/unmature/')
    if request.method=='POST':

        

        ### MATURE
        mature_images = os.listdir(mature_dir)
        total_images_mature = len(mature_images)

        # Definindo os tamanhos dos conjuntos de treinamento, validação e teste
        train_size_mature = int(0.8 * total_images_mature)
        val_size_mature = int(0.1 * total_images_mature)
        test_size_mature = total_images_mature - train_size_mature - val_size_mature

        # Embaralhar as imagens para garantir que a seleção seja aleatória
        # random.shuffle(mature_images)

        # Distribuindo as imagens para treinamento, validação e teste
        train_mature_images = mature_images[:train_size_mature]
        val_mature_images = mature_images[train_size_mature:train_size_mature + val_size_mature]
        test_mature_images = mature_images[train_size_mature + val_size_mature:train_size_mature + val_size_mature + test_size_mature]

        ### PARTIALLY MATURE
        partiallymature_images = os.listdir(partiallymature_dir)
        total_images_partiallymature = len(partiallymature_images)

        # Definindo os tamanhos dos conjuntos de treinamento, validação e teste
        train_size_partiallymature = int(0.8 * total_images_partiallymature)
        val_size_partiallymature = int(0.1 * total_images_partiallymature)
        test_size_partiallymature = total_images_partiallymature - train_size_partiallymature - val_size_partiallymature

        # Embaralhar as imagens para garantir que a seleção seja aleatória
        # random.shuffle(partiallymature_images)

        # Distribuindo as imagens para treinamento, validação e teste
        train_partiallymature_images = partiallymature_images[:train_size_partiallymature]
        val_partiallymature_images = partiallymature_images[train_size_partiallymature:train_size_partiallymature + val_size_partiallymature]
        test_partiallymature_images = partiallymature_images[train_size_partiallymature + val_size_partiallymature:train_size_partiallymature + val_size_partiallymature + test_size_partiallymature]


        ### UNMATURE
        unmature_images = os.listdir(unmature_dir)
        total_images_unmature = len(unmature_images)

        # Definindo os tamanhos dos conjuntos de treinamento, validação e teste
        train_size_unmature = int(0.8 * total_images_unmature)
        val_size_unmature = int(0.1 * total_images_unmature)
        test_size_unmature = total_images_unmature - train_size_unmature - val_size_unmature

        # Embaralhar as imagens para garantir que a seleção seja aleatória
        # random.shuffle(unmature_images)

        # Distribuindo as imagens para treinamento, validação e teste
        train_unmature_images = unmature_images[:train_size_unmature]
        val_unmature_images = unmature_images[train_size_unmature:train_size_unmature + val_size_unmature]
        test_unmature_images = unmature_images[train_size_unmature + val_size_unmature:train_size_unmature + val_size_unmature + test_size_unmature]
        

        """ train_size_partially = int(0.8 * len(partiallymature_images))
        val_size_partially = int(0.1 * len(partiallymature_images))
        test_size_partially = len(partiallymature_images) - train_size_partially - val_size_partially

        train_size_un = int(0.8 * len(unmature_images))
        val_size_un = int(0.1 * len(unmature_images))
        test_size_un = len(unmature_images) - train_size_un - val_size_un

        # Dividindo as imagens para treinamento, validação e teste
        train_mature_images = mature_images[:train_size_mature]
        val_mature_images = mature_images[train_size_mature:train_size_mature + val_size_mature]
        test_mature_images = mature_images[train_size_mature + val_size_mature:]

        # Dividindo as imagens para treinamento, validação e teste
        train_partiallymature_images = partiallymature_images[:train_size_partially]
        val_partiallymature_images = partiallymature_images[train_size_partially:train_size_partially + val_size_partially]
        test_partiallymature_images = partiallymature_images[train_size_partially + val_size_partially:]

        # Dividindo as imagens para treinamento, validação e teste
        train_unmature_images = unmature_images[:train_size_un]
        val_unmature_images = unmature_images[train_size_un:train_size_un + val_size_un]
        test_unmature_images = unmature_images[train_size_un + val_size_un:] """
        
        # Gerando dados 
        train_dir = 'train_data/'
        val_dir = 'val_data/'
        test_dir = 'test_data/'
        try:
            os.makedirs(os.path.join(BASE_DIR, train_dir, 'mature/'))
        except:
            pass
        try:
            os.makedirs(os.path.join(BASE_DIR, train_dir, 'partiallymature/'))
        except:
            pass
        try:
            os.makedirs(os.path.join(BASE_DIR, train_dir, 'unmature/'))
        except:
            pass

        try:
            os.makedirs(os.path.join(BASE_DIR, val_dir, 'mature/'))
        except:
            pass
        try:
            os.makedirs(os.path.join(BASE_DIR, val_dir, 'partiallymature/'))
        except:
            pass
        try:
            os.makedirs(os.path.join(BASE_DIR, val_dir, 'unmature/'))
        except:
            pass

        try:
            os.makedirs(os.path.join(BASE_DIR, test_dir, 'mature/'))
        except:
            pass
        try:
            os.makedirs(os.path.join(BASE_DIR, test_dir, 'partiallymature/'))
        except:
            pass
        try:
            os.makedirs(os.path.join(BASE_DIR, test_dir, 'unmature/'))
        except:
            pass


        for image in train_mature_images:
            src = mature_dir + image
            dst = os.path.join(BASE_DIR, train_dir, 'mature/')
            shutil.copy(src, dst)

        for image in train_partiallymature_images:
            src = partiallymature_dir + image
            dst = os.path.join(BASE_DIR, train_dir, 'partiallymature/')
            shutil.copy(src, dst)

        for image in train_unmature_images:
            src = unmature_dir + image
            dst = os.path.join(BASE_DIR, train_dir, 'unmature/')
            shutil.copy(src, dst)

        for image in val_mature_images:
            src = mature_dir + image
            dst = os.path.join(BASE_DIR, val_dir, 'mature/')
            shutil.copy(src, dst)

        for image in val_partiallymature_images:
            src = partiallymature_dir + image
            dst = os.path.join(BASE_DIR, val_dir, 'partiallymature/')
            shutil.copy(src, dst)

        for image in val_unmature_images:
            src = unmature_dir + image
            dst = os.path.join(BASE_DIR, val_dir, 'unmature/')
            shutil.copy(src, dst)

        for image in test_mature_images:
            src = mature_dir + image
            dst = os.path.join(BASE_DIR, test_dir, 'mature/')
            shutil.copy(src, dst)

        for image in test_partiallymature_images:
            src = partiallymature_dir + image
            dst = os.path.join(BASE_DIR, test_dir, 'partiallymature/')
            shutil.copy(src, dst)

        for image in test_unmature_images:
            src = unmature_dir + image
            dst = os.path.join(BASE_DIR, test_dir, 'unmature/')
            shutil.copy(src, dst)
        
        

        train_data_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale = 1/255.)
        val_data_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale = 1/255.)
        test_data_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale = 1/255.)

        train_dataset = train_data_gen.flow_from_directory(
            os.path.join(BASE_DIR, train_dir),
            target_size = (227,227),
            class_mode = 'categorical'
        )

        val_dataset = val_data_gen.flow_from_directory(
            os.path.join(BASE_DIR, val_dir),
            target_size = (227,227),
            class_mode = 'categorical'
        )

        test_dataset = test_data_gen.flow_from_directory(
            os.path.join(BASE_DIR, test_dir),
            target_size = (227,227),
            class_mode = 'categorical'
        )

        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(227, 227, 3)))
        model.add(tf.keras.layers.MaxPooling2D((2, 2)))
        model.add(tf.keras.layers.Flatten())

        model.add(tf.keras.layers.Dense(64, activation='relu'))
        model.add(tf.keras.layers.Dense(3, activation='softmax'))

        opt = tf.keras.optimizers.Adam(learning_rate=0.0001)

        model.summary()
        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])


        #early_stopping = tf.keras.callbacks.EarlyStopping(monitor = 'val_accuracy', mode = 'max', patience = 15)

        history = model.fit(train_dataset, epochs=5, validation_data=val_dataset)
        
        plt.plot(history.history['loss'], label = 'loss')
        plt.plot(history.history['val_loss'], label = 'val_loss')
        plt.title("Função de perda")
        plt.xlabel('Épocas')
        plt.ylabel('MSE')
        plt.legend(["Treinando"], loc='upper left')
        plt.savefig('my_figure4.png')

        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        import numpy as np
        # Fazendo previsões no conjunto de validação
        #predictions = model.predict(val_dataset)
        #predicted_classes = np.argmax(predictions, axis=-1)
        output_model = np.argmax(model.predict(test_dataset), axis=-1)
        #y_test_class =np.argmax(test_dataset.classes, axis=-1)

        # Obtendo as classes verdadeiras do conjunto de validação
        true_classes = test_dataset.classes
        print("P Classes", output_model)
        print("T Classes ", true_classes)

        # Calculando as métricas de avaliação
        accuracy = accuracy_score(true_classes, output_model)
        precision = precision_score(true_classes, output_model, average='macro')
        recall = recall_score(true_classes, output_model, average='macro')
        f1 = f1_score(true_classes, output_model, average='macro')

        print("Acurácia:", accuracy)
        print("Precisão:", precision)
        print("Sensibilidade:", recall)
        print("F1-Score:", f1)

        from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
        cm = confusion_matrix(true_classes, output_model)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        disp.ax_.set_title('Matriz de Confusão')
        disp.ax_.set_xlabel('Classificação Prevista')
        disp.ax_.set_ylabel('Classificação Real')
        plt.savefig('confusion_matrix4.png')

        # Carrega o modelo treinado
        model.save(os.path.join(BASE_DIR,'model_train.h5'))

        mensagem.append(f"O modelo foi treinado com sucesso.")
        
    context = {
        "arquivo": arquivo,
        "mensagem": mensagem,
        
    }

    return render(
        request,
        'home/index_treinamento.html',
        context,
    )