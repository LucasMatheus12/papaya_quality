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

        mature_images = os.listdir(mature_dir)
        partiallymature_images = os.listdir(partiallymature_dir)
        unmature_images = os.listdir(unmature_dir)
        train_mature_images = mature_images[:int(.8*(len(mature_images)))]
        val_mature_images = mature_images[int(.8*(len(mature_images))):]

        train_partiallymature_images = partiallymature_images[:int(.8*(len(partiallymature_images)))]
        val_partiallymature_images = partiallymature_images[int(.8*(len(partiallymature_images))):]

        train_unmature_images = unmature_images[:int(.8*(len(unmature_images)))]
        val_unmature_images = unmature_images[int(.8*(len(unmature_images))):]
        # Gerando dados 
        train_dir = 'train_data/'
        val_dir = 'val_data/'
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
        
        

        train_data_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale = 1/255.)
        val_data_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale = 1/255.)

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
        plt.savefig('my_figure2.png')

        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        import numpy as np
        # Fazendo previsões no conjunto de validação
        #predictions = model.predict(val_dataset)
        #predicted_classes = np.argmax(predictions, axis=-1)
        output_model = np.argmax(model.predict(val_dataset), axis=-1)
        y_test_class =np.argmax(model.predict(val_dataset), axis=-1)

        # Obtendo as classes verdadeiras do conjunto de validação
        true_classes = val_dataset.classes
        print("P Classes", output_model)
        print("T Classes ", y_test_class)

        # Calculando as métricas de avaliação
        accuracy = accuracy_score(y_test_class, output_model)
        precision = precision_score(y_test_class, output_model, average='macro')
        recall = recall_score(y_test_class, output_model, average='macro')
        f1 = f1_score(y_test_class, output_model, average='macro')

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
        plt.savefig('confusion_matrix.png')

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