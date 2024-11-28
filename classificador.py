import os
from PIL import Image, UnidentifiedImageError

def verificar_e_deletar_imagens_corrompidas(diretorio_base, classes):
    """
    Verifica as imagens em um diretório base com subpastas para cada classe
    e deleta as imagens corrompidas.

    Args:
        diretorio_base: O caminho para o diretório base contendo as subpastas das classes.
        classes: Uma lista de nomes de classes (subpastas).
    """

    for classe in classes:
        diretorio_classe = os.path.join(diretorio_base, classe)
        
        if not os.path.exists(diretorio_classe):
            print(f"Aviso: Diretório da classe '{classe}' não encontrado: {diretorio_classe}")
            continue
            
        for nome_arquivo in os.listdir(diretorio_classe):
            caminho_arquivo = os.path.join(diretorio_classe, nome_arquivo)
            if os.path.isfile(caminho_arquivo):
                try:
                    with Image.open(caminho_arquivo) as img:
                        pass
                except UnidentifiedImageError:
                    print(f"Aviso: Deletando imagem corrompida: {caminho_arquivo}")
                    os.remove(caminho_arquivo)
                except Exception as e:
                    print(f"Erro ao processar a imagem {caminho_arquivo}: {e}")

diretorio_base = 'PetImages'
classes = ['Cat', 'Dog']
verificar_e_deletar_imagens_corrompidas(diretorio_base, classes)


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator


train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

validation_datagen = ImageDataGenerator(rescale=1./255)


train_generator = train_datagen.flow_from_directory(
    './PetImages',
    target_size=(64, 64),
    batch_size=16,
    class_mode='binary',
    classes=['Cat', 'Dog']
)

validation_generator = validation_datagen.flow_from_directory(
    './PetImages',
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary',
    classes=['Cat', 'Dog']
)


base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(64, 64, 3))


for layer in base_model.layers:
    layer.trainable = False


x = base_model.output
x = Flatten()(x)
x = Dense(256, activation='relu')(x)
predictions = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=predictions)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=3,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size
)

loss, accuracy = model.evaluate(validation_generator, verbose=0)
print(f"Loss: {loss:.4f}")
print(f"Accuracy: {accuracy:.4f}")
