"""
Teferi_Samuel_Laforga.py
Práctica 2: Visión artificial y aprendizaje
Implementa tareas A–L según enunciado.
"""
# Supresión de logs innecesarios de TensorFlow
import logging, os
logging.disable(logging.WARNING)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers, regularizers, optimizers, applications
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from PIL import Image
from time import time

# --------------------------------------------------
# Función de carga y preprocesado de CIFAR-10
# --------------------------------------------------

def cargar_y_preprocesar_cifar10():
    (X_train, Y_train), (X_test, Y_test) = keras.datasets.cifar10.load_data()
    X_train = X_train.astype('float32') / 255.0
    X_test  = X_test.astype('float32')  / 255.0
    Y_train = keras.utils.to_categorical(Y_train, 10)
    Y_test  = keras.utils.to_categorical(Y_test,  10)
    return X_train, Y_train, X_test, Y_test

# --------------------------------------------------
# Visualización de imágenes aleatorias
# --------------------------------------------------

def mostrar_imagenes_aleatorias(X, Y, n=3):
    idx = np.random.choice(len(X), n, replace=False)
    for i in idx:
        plt.figure(); plt.imshow(X[i]); plt.title(f'Etiqueta: {np.argmax(Y[i])}'); plt.axis('off')
    plt.show()

# --------------------------------------------------
# Tarea A: MLP básico
# --------------------------------------------------

def probar_MLP(X_train, Y_train, X_test, Y_test,
               ocultas=[32], activ=['sigmoid'],
               epochs=10, batch_size=32, validation_split=0.1):
    model = keras.Sequential([keras.Input(shape=X_train.shape[1:]), layers.Flatten()])
    for u, a in zip(ocultas, activ): model.add(layers.Dense(u, activation=a))
    model.add(layers.Dense(10, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size,
                        validation_split=validation_split, verbose=2)
    plt.figure(); plt.plot(history.history['val_loss'], label='val_loss'); plt.legend(); plt.title('A MLP')
    plt.show()
    loss, acc = model.evaluate(X_test, Y_test, verbose=0)
    print(f'Tarea A → loss: {loss:.4f}, acc: {acc:.4f}')
    return model, history

# --------------------------------------------------
# Tarea B: Efecto de epochs
# --------------------------------------------------

def tarea_B_analizar_epochs(X_train, Y_train, X_test, Y_test,
                             epoch_list=[5,10,20,50]):
    for e in epoch_list:
        print(f'→ epochs={e}')
        probar_MLP(X_train, Y_train, X_test, Y_test, epochs=e)

# --------------------------------------------------
# Tarea C: Efecto de batch size
# --------------------------------------------------

def tarea_C_analizar_batch_size(X_train, Y_train, X_test, Y_test,
                                batch_list=[16,32,64,128]):
    for b in batch_list:
        print(f'→ batch_size={b}')
        probar_MLP(X_train, Y_train, X_test, Y_test, batch_size=b)

# --------------------------------------------------
# Tarea D: Optimizadores
# --------------------------------------------------

def tarea_D_explorar_optimizadores(X_train, Y_train, X_test, Y_test,
                                   optim_list=['sgd','adam','rmsprop']):
    for opt in optim_list:
        print(f'→ Optimizer={opt}')
        model = keras.Sequential([keras.Input(shape=X_train.shape[1:]), layers.Flatten(), layers.Dense(32, activation='relu'), layers.Dense(10, activation='softmax')])
        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
        history = model.fit(X_train, Y_train, epochs=10, batch_size=32, validation_split=0.1, verbose=0)
        loss, acc = model.evaluate(X_test, Y_test, verbose=0)
        print(f'   Test acc={acc:.4f}')

# --------------------------------------------------
# Tarea E: Regularización L2
# --------------------------------------------------

def tarea_E_regularizacion(X_train, Y_train, X_test, Y_test,
                            l2_list=[0.001, 0.01, 0.1]):
    """
    Prueba regularización L2 con distintos coeficientes.
    """
    for l2 in l2_list:
        print(f'→ L2={l2}')
        model = keras.Sequential([
            keras.Input(shape=X_train.shape[1:]),
            layers.Flatten(),
            layers.Dense(
                64,
                activation='relu',
                kernel_regularizer=regularizers.l2(float(l2))
            ),
            layers.Dense(10, activation='softmax')
        ])

        # Compilación nombrando cada parámetro para evitar errores de dtype
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        # Entrenamiento silencioso
        model.fit(
            X_train, Y_train,
            epochs=10,
            validation_split=0.1,
            verbose=0
        )

        # Evaluación final
        loss, acc = model.evaluate(X_test, Y_test, verbose=0)
        print(f'   Test acc={acc:.4f}')


# --------------------------------------------------
# Tarea F: Dropout
# --------------------------------------------------

def tarea_F_dropout(X_train, Y_train, X_test, Y_test,
                     rate_list=[0.2,0.4,0.6]):
    """
    Prueba diferentes tasas de Dropout y mide accuracy en test.
    """
    for r in rate_list:
        print(f'→ Dropout={r}')
        model = keras.Sequential([
            keras.Input(shape=X_train.shape[1:]),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dropout(float(r)),
            layers.Dense(10, activation='softmax')
        ])
        # Compilación explícita para evitar problemas de dtype
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        # Entrenamiento silencioso
        model.fit(
            X_train, Y_train,
            epochs=10,
            validation_split=0.1,
            verbose=0
        )
        # Evaluación
        loss, acc = model.evaluate(X_test, Y_test, verbose=0)
        print(f'   Test acc={acc:.4f}')


# --------------------------------------------------
# Tarea G: Aumentación de datos
# --------------------------------------------------
def tarea_G_aumentacion_datos(X_train, Y_train,
                              epochs=10, batch_size=32):
    """
    Aplica aumentación de datos y entrena un modelo sencillo.
    """
    # Configurar el generador de imágenes con augmentación
    datagen = keras.preprocessing.image.ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True
    )

    # Definir modelo básico
    model = keras.Sequential([
        keras.Input(shape=X_train.shape[1:]),
        layers.Flatten(),
        layers.Dense(32, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])

    # Compilar especificando nombres de parámetros
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Ajustar el generador a los datos de entrada
    datagen.fit(X_train)

    # Entrenar usando el flujo augmentado
    model.fit(
        datagen.flow(X_train, Y_train, batch_size=batch_size),
        epochs=epochs,
        verbose=2
    )

    # Evaluar sobre el conjunto de entrenamiento (o test si prefieres)
    loss, acc = model.evaluate(X_train, Y_train, verbose=0)
    print(f'Tarea G → Train acc: {acc:.4f}')


# --------------------------------------------------
# Tarea H: Transfer Learning con MobileNetV2
# --------------------------------------------------

def tarea_H_transfer_learning(X_train, Y_train, X_test, Y_test,
                              epochs=5, batch_size=32):
    """
    Aplica transfer learning con MobileNetV2:
    - Redimensiona de 32×32 a 224×224
    - Usa pesos 'imagenet' en la base (congelada)
    - Entrena solo la capa final
    """
    print("→ Tarea H: Transfer Learning")

    # 1) Modelo de preprocesado + MobileNetV2
    inputs = keras.Input(shape=X_train.shape[1:], name="input")
    x = layers.Resizing(224, 224, name="resize")(inputs)
    x = applications.mobilenet_v2.preprocess_input(x)
    base = applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet",
        pooling="avg",
        name="base_mobilenet"
    )
    base.trainable = False
    x = base(x)
    outputs = layers.Dense(10, activation="softmax", name="predictions")(x)

    model = keras.Model(inputs, outputs, name="TF_MobileNetV2")
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    # 2) Entrenamiento
    history = model.fit(
        X_train, Y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        verbose=2
    )

    # 3) Evaluación final
    loss, acc = model.evaluate(X_test, Y_test, verbose=0)
    print(f"Tarea H → Transfer acc: {acc:.4f}")

    return model, history


# --------------------------------------------------
# Tarea I: Matriz de confusión
# --------------------------------------------------

def tarea_I_confusion_matrix(model, X_test, Y_test):
    Y_pred = model.predict(X_test)
    cm = confusion_matrix(np.argmax(Y_test, axis=1), np.argmax(Y_pred, axis=1))
    disp = ConfusionMatrixDisplay(cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.show()

# --------------------------------------------------
# Tarea J: Evaluación sobre dataset propio
# --------------------------------------------------

def evaluar_dataset_propio(root_dir, model):
    X, Y = [], []
    for idx, cls in enumerate(sorted(os.listdir(root_dir))):
        for img_name in os.listdir(os.path.join(root_dir,cls)):
            img = Image.open(os.path.join(root_dir,cls,img_name)).resize((32,32))
            X.append(np.array(img)/255.0); Y.append(idx)
    X=np.stack(X); Y=keras.utils.to_categorical(Y,len(np.unique(Y)))
    print(model.evaluate(X,Y,verbose=0))

# --------------------------------------------------
# Main
# --------------------------------------------------
if __name__ == '__main__':
    X_train, Y_train, X_test, Y_test = cargar_y_preprocesar_cifar10()
    mostrar_imagenes_aleatorias(X_train, Y_train,3)
    modelA,_= probar_MLP(X_train, Y_train, X_test, Y_test)
    tarea_B_analizar_epochs(X_train, Y_train, X_test, Y_test)
    tarea_C_analizar_batch_size(X_train, Y_train, X_test, Y_test)
    tarea_D_explorar_optimizadores(X_train, Y_train, X_test, Y_test)
    tarea_E_regularizacion(X_train, Y_train, X_test, Y_test)
    tarea_F_dropout(X_train, Y_train, X_test, Y_test)
    tarea_G_aumentacion_datos(X_train, Y_train)
    tarea_H_transfer_learning(X_train, Y_train, X_test, Y_test)
    tarea_I_confusion_matrix(modelA, X_test, Y_test)
    evaluar_dataset_propio('dataset_propio', modelA)
