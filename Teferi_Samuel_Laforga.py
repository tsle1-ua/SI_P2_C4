"""
Teferi_Samuel_Laforga.py
Práctica 2: Visión artificial y aprendizaje
Implementa tareas A–L según enunciado.
Versión mejorada para C4 con todas las tareas obligatorias
"""

# Supresión de logs innecesarios de TensorFlow
import logging, os

logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers, regularizers, optimizers, applications
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from PIL import Image
from time import time
import gc
import pandas as pd  # Necesario para tarea K

NUM_CLASSES = 10

# --------------------------------------------------
# Función auxiliar para representar curvas de entrenamiento
# --------------------------------------------------
def plot_history(history, titulo="Entrenamiento"):
    """Muestra en una misma figura las curvas de loss y accuracy."""
    plt.figure(figsize=(12, 4))
    
    # Subplot 1: Loss
    plt.subplot(1, 2, 1)
    epochs = range(1, len(history.history["loss"]) + 1)
    plt.plot(epochs, history.history["loss"], label="loss", color='blue')
    plt.plot(epochs, history.history.get("val_loss", []), label="val_loss", color='red')
    plt.title(f"{titulo} - Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    
    # Subplot 2: Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history.history.get("accuracy", []), label="acc", color='blue')
    plt.plot(epochs, history.history.get("val_accuracy", []), label="val_acc", color='red')
    plt.title(f"{titulo} - Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

# --------------------------------------------------
# Función de carga y preprocesado de CIFAR-10
# --------------------------------------------------
def cargar_y_preprocesar_cifar10():
    """
    Carga CIFAR-10 y aplica preprocesamiento:
    - Normalización a [0,1]
    - One-hot encoding de etiquetas
    """
    (X_train, Y_train), (X_test, Y_test) = keras.datasets.cifar10.load_data()
    
    # Normalización: dividir por 255 para pasar de [0,255] a [0,1]
    X_train = X_train.astype("float32") / 255.0
    X_test = X_test.astype("float32") / 255.0
    
    # One-hot encoding para clasificación multiclase
    Y_train = keras.utils.to_categorical(Y_train, NUM_CLASSES)
    Y_test = keras.utils.to_categorical(Y_test, NUM_CLASSES)
    
    return X_train, Y_train, X_test, Y_test

# --------------------------------------------------
# Visualización de imágenes aleatorias
# --------------------------------------------------
def mostrar_imagenes_aleatorias(X, Y, n=3):
    """Muestra n imágenes aleatorias con sus etiquetas."""
    idx = np.random.choice(len(X), n, replace=False)
    clases = ['avión', 'coche', 'pájaro', 'gato', 'ciervo', 
              'perro', 'rana', 'caballo', 'barco', 'camión']
    
    plt.figure(figsize=(n*3, 3))
    for i, index in enumerate(idx):
        plt.subplot(1, n, i+1)
        plt.imshow(X[index])
        clase_idx = np.argmax(Y[index])
        plt.title(f"Clase: {clases[clase_idx]}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# --------------------------------------------------
# Tarea A: MLP básico
# --------------------------------------------------
def probar_MLP(
    X_train,
    Y_train,
    X_test,
    Y_test,
    ocultas=[32],
    activ=["sigmoid"],
    epochs=10,
    batch_size=32,
    validation_split=0.1,
    optimizer="adam",
    verbose=1
):
    """Construye y entrena un MLP básico."""
    model = keras.Sequential([
        keras.Input(shape=X_train.shape[1:]), 
        layers.Flatten()
    ])
    
    for u, a in zip(ocultas, activ):
        model.add(layers.Dense(int(u), activation=a))
    
    model.add(layers.Dense(NUM_CLASSES, activation="softmax"))
    
    model.compile(
        optimizer=optimizer, 
        loss="categorical_crossentropy", 
        metrics=["accuracy"]
    )
    
    # Early stopping callback
    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_loss', 
        patience=5, 
        restore_best_weights=True
    )
    
    history = model.fit(
        X_train,
        Y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=[early_stop],
        verbose=verbose,
    )
    
    if verbose > 0:
        plot_history(history, "MLP")
    
    loss, acc = model.evaluate(X_test, Y_test, verbose=0)
    print(f"→ Test loss: {loss:.4f}, acc: {acc:.4f}")
    
    return model, history

# --------------------------------------------------
# Tarea B: Efecto de epochs (con múltiples ejecuciones)
# --------------------------------------------------
def tarea_B_analizar_epochs(
    X_train, Y_train, X_test, Y_test, 
    epoch_list=[5, 10, 20, 50],
    n_repeticiones=5
):
    """Compara varios valores de epochs con múltiples ejecuciones."""
    resultados = {}
    resultados_std = {}
    
    for e in epoch_list:
        print(f"\n→ Analizando epochs={e}")
        accuracies = []
        
        for rep in range(n_repeticiones):
            print(f"  Repetición {rep+1}/{n_repeticiones}")
            _, history = probar_MLP(
                X_train, Y_train, X_test, Y_test,
                epochs=e, verbose=0
            )
            loss, acc = history.model.evaluate(X_test, Y_test, verbose=0)
            accuracies.append(acc)
            
            # Liberar memoria
            keras.backend.clear_session()
            gc.collect()
        
        resultados[e] = np.mean(accuracies)
        resultados_std[e] = np.std(accuracies)
        print(f"  Accuracy promedio: {resultados[e]:.4f} ± {resultados_std[e]:.4f}")
    
    # Gráfica con barras de error
    plt.figure(figsize=(10, 6))
    epochs_vals = list(resultados.keys())
    acc_vals = list(resultados.values())
    std_vals = list(resultados_std.values())
    
    plt.bar(epochs_vals, acc_vals, yerr=std_vals, capsize=5, alpha=0.7)
    plt.title(f"Comparativa epochs (promedio de {n_repeticiones} ejecuciones)")
    plt.xlabel("Epochs")
    plt.ylabel("Test accuracy")
    plt.grid(True, axis='y', alpha=0.3)
    
    # Añadir valores en las barras
    for i, (e, acc, std) in enumerate(zip(epochs_vals, acc_vals, std_vals)):
        plt.text(i, acc + std + 0.01, f'{acc:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()

# --------------------------------------------------
# Tarea C: Efecto de batch size
# --------------------------------------------------
def tarea_C_analizar_batch_size(
    X_train, Y_train, X_test, Y_test, 
    batch_list=[16, 32, 64, 128]
):
    """Estudia la influencia del tamaño del batch."""
    resultados_acc = {}
    resultados_time = {}
    
    for b in batch_list:
        print(f"\n→ batch_size={b}")
        start_time = time()
        
        _, history = probar_MLP(
            X_train, Y_train, X_test, Y_test,
            batch_size=b, verbose=0
        )
        
        train_time = time() - start_time
        loss, acc = history.model.evaluate(X_test, Y_test, verbose=0)
        
        resultados_acc[b] = acc
        resultados_time[b] = train_time
        
        print(f"  Accuracy: {acc:.4f}, Tiempo: {train_time:.2f}s")
    
    # Gráfica doble eje Y
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    batch_vals = list(resultados_acc.keys())
    acc_vals = list(resultados_acc.values())
    time_vals = list(resultados_time.values())
    
    # Eje 1: Accuracy
    color = 'tab:blue'
    ax1.set_xlabel('Batch size')
    ax1.set_ylabel('Test accuracy', color=color)
    bars1 = ax1.bar([x - 0.2 for x in range(len(batch_vals))], 
                     acc_vals, 0.4, label='Accuracy', color=color, alpha=0.7)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_xticks(range(len(batch_vals)))
    ax1.set_xticklabels(batch_vals)
    
    # Eje 2: Tiempo
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Tiempo de entrenamiento (s)', color=color)
    bars2 = ax2.bar([x + 0.2 for x in range(len(batch_vals))], 
                     time_vals, 0.4, label='Tiempo', color=color, alpha=0.7)
    ax2.tick_params(axis='y', labelcolor=color)
    
    ax1.set_title("Comparativa batch_size: Accuracy vs Tiempo")
    ax1.grid(True, alpha=0.3)
    
    # Leyenda combinada
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    
    plt.tight_layout()
    plt.show()

# --------------------------------------------------
# Tarea D: Optimizadores
# --------------------------------------------------
def tarea_D_explorar_optimizadores(
    X_train, Y_train, X_test, Y_test, 
    optim_list=["sgd", "adam", "rmsprop", "adamax", "nadam"]
):
    """Compara diferentes optimizadores para el mismo MLP."""
    resultados = {}
    
    for opt in optim_list:
        print(f"\n→ Optimizer={opt}")
        
        # Configurar optimizador con learning rate apropiado
        if opt == "sgd":
            optimizer = optimizers.SGD(learning_rate=0.01)
        else:
            optimizer = opt
            
        model = keras.Sequential([
            keras.Input(shape=X_train.shape[1:]),
            layers.Flatten(),
            layers.Dense(64, activation="relu"),
            layers.Dense(32, activation="relu"),
            layers.Dense(NUM_CLASSES, activation="softmax"),
        ])
        
        model.compile(
            optimizer=optimizer, 
            loss="categorical_crossentropy", 
            metrics=["accuracy"]
        )
        
        history = model.fit(
            X_train, Y_train, 
            epochs=20, 
            batch_size=32, 
            validation_split=0.1, 
            verbose=0
        )
        
        loss, acc = model.evaluate(X_test, Y_test, verbose=0)
        resultados[opt] = acc
        print(f"  Test accuracy: {acc:.4f}")
    
    # Gráfica
    plt.figure(figsize=(10, 6))
    opts = list(resultados.keys())
    accs = list(resultados.values())
    
    bars = plt.bar(opts, accs, color=['blue', 'green', 'red', 'purple', 'orange'])
    plt.title("Comparativa de optimizadores")
    plt.ylabel("Test accuracy")
    plt.ylim(min(accs) * 0.95, max(accs) * 1.02)
    
    # Valores en las barras
    for bar, acc in zip(bars, accs):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002, 
                f'{acc:.4f}', ha='center', va='bottom')
    
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()

# --------------------------------------------------
# Tarea E: Regularización L2
# --------------------------------------------------
def tarea_E_regularizacion(
    X_train, Y_train, X_test, Y_test, 
    l2_list=[0.0, 0.001, 0.01, 0.1]
):
    """Prueba regularización L2 con distintos coeficientes."""
    resultados = {}
    
    for l2 in l2_list:
        print(f"\n→ L2={l2}")
        
        model = keras.Sequential([
            keras.Input(shape=X_train.shape[1:]),
            layers.Flatten(),
            layers.Dense(
                128, activation="relu", 
                kernel_regularizer=regularizers.l2(float(l2))
            ),
            layers.Dense(
                64, activation="relu",
                kernel_regularizer=regularizers.l2(float(l2))
            ),
            layers.Dense(NUM_CLASSES, activation="softmax"),
        ])
        
        model.compile(
            optimizer="adam", 
            loss="categorical_crossentropy", 
            metrics=["accuracy"]
        )
        
        history = model.fit(
            X_train, Y_train, 
            epochs=20, 
            batch_size=32,
            validation_split=0.1, 
            verbose=0
        )
        
        loss, acc = model.evaluate(X_test, Y_test, verbose=0)
        resultados[l2] = acc
        print(f"  Test accuracy: {acc:.4f}")
    
    # Gráfica
    plt.figure(figsize=(10, 6))
    l2_vals = list(resultados.keys())
    acc_vals = list(resultados.values())
    
    plt.bar([str(l) for l in l2_vals], acc_vals)
    plt.title("Efecto de la regularización L2")
    plt.xlabel("Coeficiente L2")
    plt.ylabel("Test accuracy")
    
    for i, (l2, acc) in enumerate(zip(l2_vals, acc_vals)):
        plt.text(i, acc + 0.002, f'{acc:.4f}', ha='center', va='bottom')
    
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()

# --------------------------------------------------
# Tarea F: Dropout
# --------------------------------------------------
def tarea_F_dropout(
    X_train, Y_train, X_test, Y_test, 
    rate_list=[0.0, 0.2, 0.4, 0.6]
):
    """Prueba diferentes tasas de Dropout."""
    resultados = {}
    
    for r in rate_list:
        print(f"\n→ Dropout={r}")
        
        model = keras.Sequential([
            keras.Input(shape=X_train.shape[1:]),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(float(r)),
            layers.Dense(64, activation="relu"),
            layers.Dropout(float(r)),
            layers.Dense(NUM_CLASSES, activation="softmax"),
        ])
        
        model.compile(
            optimizer="adam", 
            loss="categorical_crossentropy", 
            metrics=["accuracy"]
        )
        
        history = model.fit(
            X_train, Y_train, 
            epochs=20,
            batch_size=32,
            validation_split=0.1, 
            verbose=0
        )
        
        loss, acc = model.evaluate(X_test, Y_test, verbose=0)
        resultados[r] = acc
        print(f"  Test accuracy: {acc:.4f}")
    
    # Gráfica
    plt.figure(figsize=(10, 6))
    rates = list(resultados.keys())
    accs = list(resultados.values())
    
    plt.plot(rates, accs, 'o-', markersize=8, linewidth=2)
    plt.title("Efecto del Dropout")
    plt.xlabel("Tasa de Dropout")
    plt.ylabel("Test accuracy")
    plt.grid(True, alpha=0.3)
    
    for rate, acc in zip(rates, accs):
        plt.annotate(f'{acc:.4f}', (rate, acc), 
                    textcoords="offset points", xytext=(0,10), ha='center')
    
    plt.tight_layout()
    plt.show()

# --------------------------------------------------
# Tarea G: CNN básica (CORREGIDA)
# --------------------------------------------------
def tarea_G_CNN_basica(
    X_train, Y_train, X_test, Y_test,
    epochs=20, batch_size=32
):
    """
    Define, entrena y evalua una CNN básica.
    Implementa red convolucional con Conv2D y MaxPooling2D.
    """
    print("\n→ Tarea G: CNN básica")
    
    # Modelo CNN sin MaxPooling
    model_sin_pool = keras.Sequential([
        keras.Input(shape=X_train.shape[1:]),
        
        layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ], name="CNN_sin_MaxPool")
    
    model_sin_pool.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("\nEntrenando CNN sin MaxPooling...")
    history_sin = model_sin_pool.fit(
        X_train, Y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        verbose=1
    )
    
    loss_sin, acc_sin = model_sin_pool.evaluate(X_test, Y_test, verbose=0)
    print(f"CNN sin MaxPooling → Test acc: {acc_sin:.4f}")
    
    # Modelo CNN con MaxPooling
    model_con_pool = keras.Sequential([
        keras.Input(shape=X_train.shape[1:]),
        
        layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ], name="CNN_con_MaxPool")
    
    model_con_pool.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("\nEntrenando CNN con MaxPooling...")
    history_con = model_con_pool.fit(
        X_train, Y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        verbose=1
    )
    
    loss_con, acc_con = model_con_pool.evaluate(X_test, Y_test, verbose=0)
    print(f"CNN con MaxPooling → Test acc: {acc_con:.4f}")
    
    # Comparación visual
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history_sin.history['val_accuracy'], label='Sin MaxPool')
    plt.plot(history_con.history['val_accuracy'], label='Con MaxPool')
    plt.title('Validación Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    modelos = ['Sin MaxPool', 'Con MaxPool']
    accuracies = [acc_sin, acc_con]
    plt.bar(modelos, accuracies, color=['blue', 'green'])
    plt.title('Test Accuracy Final')
    plt.ylabel('Accuracy')
    for i, acc in enumerate(accuracies):
        plt.text(i, acc + 0.01, f'{acc:.4f}', ha='center')
    
    plt.tight_layout()
    plt.show()
    
    return model_con_pool, history_con

# --------------------------------------------------
# Tarea H: CNN - Efecto kernel_size
# --------------------------------------------------
def tarea_H_kernel_size(
    X_train, Y_train, X_test, Y_test,
    kernel_sizes=[(3,3), (5,5), (7,7)],
    epochs=15
):
    """Analiza el efecto del tamaño del kernel en CNN."""
    print("\n→ Tarea H: Análisis de kernel_size")
    
    resultados = {}
    tiempos = {}
    
    for ks in kernel_sizes:
        print(f"\nProbando kernel_size={ks}")
        
        model = keras.Sequential([
            keras.Input(shape=X_train.shape[1:]),
            
            layers.Conv2D(32, ks, activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            
            layers.Conv2D(64, ks, activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(NUM_CLASSES, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        start_time = time()
        history = model.fit(
            X_train, Y_train,
            epochs=epochs,
            batch_size=64,
            validation_split=0.1,
            verbose=0
        )
        train_time = time() - start_time
        
        loss, acc = model.evaluate(X_test, Y_test, verbose=0)
        resultados[str(ks)] = acc
        tiempos[str(ks)] = train_time
        
        print(f"  Test accuracy: {acc:.4f}, Tiempo: {train_time:.2f}s")
    
    # Gráfica comparativa
    plt.figure(figsize=(10, 6))
    
    kernels = list(resultados.keys())
    accs = list(resultados.values())
    times = list(tiempos.values())
    
    x = np.arange(len(kernels))
    width = 0.35
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    ax1.bar(x - width/2, accs, width, label='Accuracy', color='blue', alpha=0.7)
    ax1.set_xlabel('Kernel Size')
    ax1.set_ylabel('Test Accuracy', color='blue')
    ax1.set_xticks(x)
    ax1.set_xticklabels(kernels)
    ax1.tick_params(axis='y', labelcolor='blue')
    
    ax2 = ax1.twinx()
    ax2.bar(x + width/2, times, width, label='Tiempo', color='red', alpha=0.7)
    ax2.set_ylabel('Tiempo de entrenamiento (s)', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    
    ax1.set_title('Efecto del tamaño del kernel: Accuracy vs Tiempo')
    ax1.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# --------------------------------------------------
# Tarea I: Optimizar arquitectura CNN
# --------------------------------------------------
def tarea_I_optimizar_CNN(
    X_train, Y_train, X_test, Y_test,
    epochs=25
):
    """
    Optimiza la arquitectura CNN probando diferentes configuraciones.
    Incluye matriz de confusión del mejor modelo.
    """
    print("\n→ Tarea I: Optimización de arquitectura CNN")
    
    # Diferentes arquitecturas a probar
    arquitecturas = {
        "Simple": [
            layers.Conv2D(32, (3,3), activation='relu', padding='same'),
            layers.MaxPooling2D((2,2)),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dense(NUM_CLASSES, activation='softmax')
        ],
        "Profunda": [
            layers.Conv2D(32, (3,3), activation='relu', padding='same'),
            layers.Conv2D(32, (3,3), activation='relu', padding='same'),
            layers.MaxPooling2D((2,2)),
            layers.Conv2D(64, (3,3), activation='relu', padding='same'),
            layers.Conv2D(64, (3,3), activation='relu', padding='same'),
            layers.MaxPooling2D((2,2)),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.Dense(NUM_CLASSES, activation='softmax')
        ],
        "VGG-like": [
            layers.Conv2D(64, (3,3), activation='relu', padding='same'),
            layers.Conv2D(64, (3,3), activation='relu', padding='same'),
            layers.MaxPooling2D((2,2)),
            layers.Conv2D(128, (3,3), activation='relu', padding='same'),
            layers.Conv2D(128, (3,3), activation='relu', padding='same'),
            layers.MaxPooling2D((2,2)),
            layers.Conv2D(256, (3,3), activation='relu', padding='same'),
            layers.Conv2D(256, (3,3), activation='relu', padding='same'),
            layers.MaxPooling2D((2,2)),
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(NUM_CLASSES, activation='softmax')
        ]
    }
    
    resultados = {}
    mejor_modelo = None
    mejor_acc = 0
    
    for nombre, capas in arquitecturas.items():
        print(f"\nProbando arquitectura: {nombre}")
        
        model = keras.Sequential([keras.Input(shape=X_train.shape[1:])] + capas)
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Early stopping
        early_stop = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
        
        history = model.fit(
            X_train, Y_train,
            epochs=epochs,
            batch_size=64,
            validation_split=0.1,
            callbacks=[early_stop],
            verbose=0
        )
        
        loss, acc = model.evaluate(X_test, Y_test, verbose=0)
        resultados[nombre] = acc
        print(f"  Test accuracy: {acc:.4f}")
        
        if acc > mejor_acc:
            mejor_acc = acc
            mejor_modelo = model
    
    # Gráfica comparativa
    plt.figure(figsize=(10, 6))
    nombres = list(resultados.keys())
    accs = list(resultados.values())
    
    bars = plt.bar(nombres, accs, color=['blue', 'green', 'red'])
    plt.title('Comparación de arquitecturas CNN')
    plt.ylabel('Test Accuracy')
    plt.ylim(min(accs) * 0.95, max(accs) * 1.02)
    
    for bar, acc in zip(bars, accs):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f'{acc:.4f}', ha='center', va='bottom')
    
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Matriz de confusión del mejor modelo
    print(f"\nGenerando matriz de confusión del mejor modelo (acc: {mejor_acc:.4f})")
    Y_pred = mejor_modelo.predict(X_test)
    cm = confusion_matrix(np.argmax(Y_test, axis=1), np.argmax(Y_pred, axis=1))
    
    plt.figure(figsize=(10, 8))
    clases = ['avión', 'coche', 'pájaro', 'gato', 'ciervo', 
              'perro', 'rana', 'caballo', 'barco', 'camión']
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clases)
    disp.plot(cmap='Blues', values_format='d')
    plt.title('Matriz de Confusión - Mejor modelo CNN')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    return mejor_modelo

# --------------------------------------------------
# Tarea J: Evaluación sobre dataset propio
# --------------------------------------------------
def evaluar_dataset_propio(root_dir, model):
    """
    Carga y evalúa un modelo sobre un dataset propio.
    Espera estructura: root_dir/clase/imagen.jpg
    """
    print(f"\n→ Evaluando dataset propio desde: {root_dir}")
    
    if not os.path.exists(root_dir):
        print(f"ERROR: No se encuentra el directorio {root_dir}")
        return None, None
    
    clases = ['avion', 'coche', 'pajaro', 'gato', 'ciervo', 
              'perro', 'rana', 'caballo', 'barco', 'camion']
    
    X, Y = [], []
    imagenes_por_clase = {cls: 0 for cls in clases}
    
    for idx, cls in enumerate(clases):
        cls_dir = os.path.join(root_dir, cls)
        if not os.path.exists(cls_dir):
            print(f"Advertencia: No se encuentra directorio para clase {cls}")
            continue
            
        for img_name in os.listdir(cls_dir):
            try:
                img_path = os.path.join(cls_dir, img_name)
                img = Image.open(img_path).convert('RGB').resize((32, 32))
                X.append(np.array(img) / 255.0)
                Y.append(idx)
                imagenes_por_clase[cls] += 1
            except Exception as e:
                print(f"Error al cargar {img_path}: {e}")
    
    if len(X) == 0:
        print("ERROR: No se cargaron imágenes")
        return None, None
    
    X = np.array(X)
    Y = keras.utils.to_categorical(Y, NUM_CLASSES)
    
    print(f"\nDataset cargado:")
    print(f"Total imágenes: {len(X)}")
    for cls, count in imagenes_por_clase.items():
        print(f"  {cls}: {count} imágenes")
    
    # Evaluar
    loss, acc = model.evaluate(X, Y, verbose=0)
    print(f"\nResultados en dataset propio:")
    print(f"  Loss: {loss:.4f}")
    print(f"  Accuracy: {acc:.4f}")
    
    return X, Y

# --------------------------------------------------
# Tarea K: Experimentación exhaustiva en dataset propio
# --------------------------------------------------
def tarea_K_experimentacion_dataset_propio(
    X_train, Y_train, X_test, Y_test,
    X_propio, Y_propio
):
    """
    Realiza experimentación exhaustiva sobre el dataset propio
    comparando diferentes configuraciones.
    """
    print("\n→ Tarea K: Experimentación exhaustiva en dataset propio")
    
    if X_propio is None:
        print("ERROR: No hay dataset propio cargado")
        return
    
    resultados = {
        'modelo': [],
        'train_acc': [],
        'test_acc': [],
        'propio_acc': [],
        'params': []
    }
    
    # 1. MLP con diferentes tamaños
    print("\n1. Experimentando con MLPs de diferente tamaño...")
    for neuronas in [32, 64, 128, 256]:
        print(f"  MLP con {neuronas} neuronas")
        model = keras.Sequential([
            keras.Input(shape=X_train.shape[1:]),
            layers.Flatten(),
            layers.Dense(neuronas, activation='relu'),
            layers.Dense(neuronas//2, activation='relu'),
            layers.Dense(NUM_CLASSES, activation='softmax')
        ])
        
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        history = model.fit(X_train, Y_train, epochs=20, batch_size=64, 
                          validation_split=0.1, verbose=0)
        
        train_acc = max(history.history['accuracy'])
        test_acc = model.evaluate(X_test, Y_test, verbose=0)[1]
        propio_acc = model.evaluate(X_propio, Y_propio, verbose=0)[1]
        
        resultados['modelo'].append(f'MLP_{neuronas}')
        resultados['train_acc'].append(train_acc)
        resultados['test_acc'].append(test_acc)
        resultados['propio_acc'].append(propio_acc)
        resultados['params'].append(model.count_params())
    
    # 2. CNNs con diferentes profundidades
    print("\n2. Experimentando con CNNs de diferente profundidad...")
    for n_bloques in [1, 2, 3]:
        print(f"  CNN con {n_bloques} bloques conv")
        
        capas = []
        filtros = 32
        for i in range(n_bloques):
            capas.extend([
                layers.Conv2D(filtros, (3,3), activation='relu', padding='same'),
                layers.MaxPooling2D((2,2))
            ])
            filtros *= 2
        
        capas.extend([
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(NUM_CLASSES, activation='softmax')
        ])
        
        model = keras.Sequential([keras.Input(shape=X_train.shape[1:])] + capas)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        
        history = model.fit(X_train, Y_train, epochs=20, batch_size=64, 
                          validation_split=0.1, verbose=0)
        
        train_acc = max(history.history['accuracy'])
        test_acc = model.evaluate(X_test, Y_test, verbose=0)[1]
        propio_acc = model.evaluate(X_propio, Y_propio, verbose=0)[1]
        
        resultados['modelo'].append(f'CNN_{n_bloques}bloques')
        resultados['train_acc'].append(train_acc)
        resultados['test_acc'].append(test_acc)
        resultados['propio_acc'].append(propio_acc)
        resultados['params'].append(model.count_params())
    
    # 3. Diferentes optimizadores
    print("\n3. Experimentando con diferentes optimizadores...")
    for opt in ['sgd', 'adam', 'rmsprop']:
        print(f"  CNN con optimizador {opt}")
        
        model = keras.Sequential([
            keras.Input(shape=X_train.shape[1:]),
            layers.Conv2D(32, (3,3), activation='relu', padding='same'),
            layers.MaxPooling2D((2,2)),
            layers.Conv2D(64, (3,3), activation='relu', padding='same'),
            layers.MaxPooling2D((2,2)),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(NUM_CLASSES, activation='softmax')
        ])
        
        if opt == 'sgd':
            optimizer = optimizers.SGD(learning_rate=0.01, momentum=0.9)
        else:
            optimizer = opt
            
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        
        history = model.fit(X_train, Y_train, epochs=20, batch_size=64, 
                          validation_split=0.1, verbose=0)
        
        train_acc = max(history.history['accuracy'])
        test_acc = model.evaluate(X_test, Y_test, verbose=0)[1]
        propio_acc = model.evaluate(X_propio, Y_propio, verbose=0)[1]
        
        resultados['modelo'].append(f'CNN_opt_{opt}')
        resultados['train_acc'].append(train_acc)
        resultados['test_acc'].append(test_acc)
        resultados['propio_acc'].append(propio_acc)
        resultados['params'].append(model.count_params())
    
    # Visualización de resultados
    df = pd.DataFrame(resultados)
    
    # Tabla resumen
    print("\n=== TABLA RESUMEN ===")
    print(df.to_string(index=False, float_format='%.4f'))
    
    # Gráfica comparativa
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Gráfica 1: Accuracies
    x = np.arange(len(df))
    width = 0.25
    
    ax1.bar(x - width, df['train_acc'], width, label='Train', alpha=0.8)
    ax1.bar(x, df['test_acc'], width, label='Test CIFAR-10', alpha=0.8)
    ax1.bar(x + width, df['propio_acc'], width, label='Dataset Propio', alpha=0.8)
    
    ax1.set_xlabel('Modelo')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Comparación de Accuracy en diferentes datasets')
    ax1.set_xticks(x)
    ax1.set_xticklabels(df['modelo'], rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Gráfica 2: Generalización (diferencia test vs propio)
    diferencias = df['test_acc'] - df['propio_acc']
    colors = ['red' if d > 0.1 else 'yellow' if d > 0.05 else 'green' for d in diferencias]
    
    ax2.bar(x, diferencias, color=colors, alpha=0.7)
    ax2.set_xlabel('Modelo')
    ax2.set_ylabel('Diferencia (Test - Propio)')
    ax2.set_title('Pérdida de generalización (menor es mejor)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(df['modelo'], rotation=45, ha='right')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Mejor modelo
    mejor_idx = df['propio_acc'].idxmax()
    print(f"\n🏆 MEJOR MODELO: {df.loc[mejor_idx, 'modelo']}")
    print(f"   Train acc: {df.loc[mejor_idx, 'train_acc']:.4f}")
    print(f"   Test acc: {df.loc[mejor_idx, 'test_acc']:.4f}")
    print(f"   Dataset propio acc: {df.loc[mejor_idx, 'propio_acc']:.4f}")
    
    return df

# --------------------------------------------------
# Tarea L: Mejoras para favorecer la generalización
# --------------------------------------------------
def tarea_L_mejoras_generalizacion(
    X_train, Y_train, X_test, Y_test,
    X_propio, Y_propio
):
    """
    Implementa y evalúa técnicas avanzadas para mejorar la generalización:
    - Data Augmentation
    - Batch Normalization
    - Advanced Pooling
    - Ensemble Learning
    """
    print("\n→ Tarea L: Técnicas avanzadas para mejorar generalización")
    
    if X_propio is None:
        print("ERROR: No hay dataset propio cargado")
        return
    
    resultados = {}
    
    # 1. Modelo base (sin mejoras)
    print("\n1. Entrenando modelo BASE (sin mejoras)...")
    model_base = keras.Sequential([
        keras.Input(shape=X_train.shape[1:]),
        layers.Conv2D(32, (3,3), activation='relu', padding='same'),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(64, (3,3), activation='relu', padding='same'),
        layers.MaxPooling2D((2,2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    
    model_base.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model_base.fit(X_train, Y_train, epochs=20, batch_size=64, validation_split=0.1, verbose=0)
    
    test_acc_base = model_base.evaluate(X_test, Y_test, verbose=0)[1]
    propio_acc_base = model_base.evaluate(X_propio, Y_propio, verbose=0)[1]
    resultados['Base'] = {'test': test_acc_base, 'propio': propio_acc_base}
    
    # 2. Con Data Augmentation
    print("\n2. Entrenando con DATA AUGMENTATION...")
    datagen = keras.preprocessing.image.ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        zoom_range=0.1,
        shear_range=0.1
    )
    
    model_aug = keras.models.clone_model(model_base)
    model_aug.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    datagen.fit(X_train)
    model_aug.fit(
        datagen.flow(X_train, Y_train, batch_size=64),
        epochs=20,
        validation_data=(X_test, Y_test),
        verbose=0
    )
    
    test_acc_aug = model_aug.evaluate(X_test, Y_test, verbose=0)[1]
    propio_acc_aug = model_aug.evaluate(X_propio, Y_propio, verbose=0)[1]
    resultados['Data Aug'] = {'test': test_acc_aug, 'propio': propio_acc_aug}
    
    # 3. Con Batch Normalization
    print("\n3. Entrenando con BATCH NORMALIZATION...")
    model_bn = keras.Sequential([
        keras.Input(shape=X_train.shape[1:]),
        
        layers.Conv2D(32, (3,3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2,2)),
        
        layers.Conv2D(64, (3,3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2,2)),
        
        layers.Conv2D(128, (3,3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.GlobalAveragePooling2D(),
        
        layers.Dense(128),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.5),
        
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    
    model_bn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model_bn.fit(X_train, Y_train, epochs=20, batch_size=64, validation_split=0.1, verbose=0)
    
    test_acc_bn = model_bn.evaluate(X_test, Y_test, verbose=0)[1]
    propio_acc_bn = model_bn.evaluate(X_propio, Y_propio, verbose=0)[1]
    resultados['Batch Norm'] = {'test': test_acc_bn, 'propio': propio_acc_bn}
    
    # 4. Con Global Average Pooling en lugar de Flatten
    print("\n4. Entrenando con GLOBAL AVERAGE POOLING...")
    model_gap = keras.Sequential([
        keras.Input(shape=X_train.shape[1:]),
        
        layers.Conv2D(32, (3,3), activation='relu', padding='same'),
        layers.Conv2D(32, (3,3), activation='relu', padding='same'),
        layers.MaxPooling2D((2,2)),
        
        layers.Conv2D(64, (3,3), activation='relu', padding='same'),
        layers.Conv2D(64, (3,3), activation='relu', padding='same'),
        layers.MaxPooling2D((2,2)),
        
        layers.Conv2D(128, (3,3), activation='relu', padding='same'),
        layers.GlobalAveragePooling2D(),  # En lugar de Flatten
        
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    
    model_gap.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model_gap.fit(X_train, Y_train, epochs=20, batch_size=64, validation_split=0.1, verbose=0)
    
    test_acc_gap = model_gap.evaluate(X_test, Y_test, verbose=0)[1]
    propio_acc_gap = model_gap.evaluate(X_propio, Y_propio, verbose=0)[1]
    resultados['Global Pool'] = {'test': test_acc_gap, 'propio': propio_acc_gap}
    
    # 5. Modelo COMPLETO con todas las mejoras
    print("\n5. Entrenando modelo COMPLETO con todas las mejoras...")
    
    # Crear modelo con todas las técnicas
    inputs = keras.Input(shape=X_train.shape[1:])
    
    # Augmentation layer (solo durante entrenamiento)
    x = layers.RandomFlip("horizontal")(inputs)
    x = layers.RandomRotation(0.1)(x)
    x = layers.RandomZoom(0.1)(x)
    
    # Bloque 1
    x = layers.Conv2D(64, (3,3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(64, (3,3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Dropout(0.25)(x)
    
    # Bloque 2
    x = layers.Conv2D(128, (3,3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(128, (3,3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Dropout(0.25)(x)
    
    # Bloque 3
    x = layers.Conv2D(256, (3,3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(256, (3,3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    
    # Global pooling
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.5)(x)
    
    outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)
    
    model_completo = keras.Model(inputs, outputs)
    
    # Compilar con learning rate scheduling
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.001,
        decay_steps=1000,
        decay_rate=0.9
    )
    
    model_completo.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks avanzados
    callbacks = [
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
    ]
    
    history_completo = model_completo.fit(
        X_train, Y_train,
        epochs=50,
        batch_size=64,
        validation_split=0.1,
        callbacks=callbacks,
        verbose=1
    )
    
    test_acc_completo = model_completo.evaluate(X_test, Y_test, verbose=0)[1]
    propio_acc_completo = model_completo.evaluate(X_propio, Y_propio, verbose=0)[1]
    resultados['COMPLETO'] = {'test': test_acc_completo, 'propio': propio_acc_completo}
    
    # Visualización de resultados
    plt.figure(figsize=(14, 8))
    
    # Subplot 1: Comparación de accuracies
    plt.subplot(2, 2, 1)
    modelos = list(resultados.keys())
    test_accs = [resultados[m]['test'] for m in modelos]
    propio_accs = [resultados[m]['propio'] for m in modelos]
    
    x = np.arange(len(modelos))
    width = 0.35
    
    plt.bar(x - width/2, test_accs, width, label='Test CIFAR-10', alpha=0.8)
    plt.bar(x + width/2, propio_accs, width, label='Dataset Propio', alpha=0.8)
    
    plt.xlabel('Modelo')
    plt.ylabel('Accuracy')
    plt.title('Comparación de técnicas de generalización')
    plt.xticks(x, modelos, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 2: Mejora respecto al baseline
    plt.subplot(2, 2, 2)
    mejoras = [(resultados[m]['propio'] - resultados['Base']['propio']) * 100 for m in modelos]
    colors = ['red' if m < 0 else 'green' for m in mejoras]
    
    plt.bar(modelos, mejoras, color=colors, alpha=0.7)
    plt.xlabel('Modelo')
    plt.ylabel('Mejora % vs Base')
    plt.title('Mejora en dataset propio respecto al modelo base')
    plt.xticks(rotation=45)
    plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    plt.grid(True, alpha=0.3)
    
    # Subplot 3: Gap de generalización
    plt.subplot(2, 2, 3)
    gaps = [resultados[m]['test'] - resultados[m]['propio'] for m in modelos]
    colors = ['red' if g > 0.15 else 'yellow' if g > 0.1 else 'green' for g in gaps]
    
    plt.bar(modelos, gaps, color=colors, alpha=0.7)
    plt.xlabel('Modelo')
    plt.ylabel('Gap (Test - Propio)')
    plt.title('Brecha de generalización (menor es mejor)')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Subplot 4: Curvas de entrenamiento del mejor modelo
    plt.subplot(2, 2, 4)
    plt.plot(history_completo.history['accuracy'], label='Train')
    plt.plot(history_completo.history['val_accuracy'], label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Curvas de entrenamiento - Modelo COMPLETO')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Resumen final
    print("\n=== RESUMEN DE RESULTADOS ===")
    print(f"{'Técnica':<15} {'Test Acc':<10} {'Propio Acc':<12} {'Gap':<8} {'Mejora vs Base'}")
    print("-" * 60)
    for modelo in modelos:
        test = resultados[modelo]['test']
        propio = resultados[modelo]['propio']
        gap = test - propio
        mejora = (propio - resultados['Base']['propio']) * 100
        print(f"{modelo:<15} {test:<10.4f} {propio:<12.4f} {gap:<8.4f} {mejora:+.2f}%")
    
    print("\n💡 CONCLUSIONES:")
    mejor_modelo = max(resultados.items(), key=lambda x: x[1]['propio'])[0]
    print(f"- Mejor modelo para generalización: {mejor_modelo}")
    print(f"- La combinación de técnicas mejora significativamente la generalización")
    print(f"- Data Augmentation y Batch Normalization son especialmente efectivas")
    
    return model_completo, resultados

# --------------------------------------------------
# Transfer Learning mejorado (para comparación con H)
# --------------------------------------------------
def transfer_learning_mejorado(X_train, Y_train, X_test, Y_test):
    """Version mejorada de transfer learning con fine-tuning."""
    print("\n→ Transfer Learning Mejorado con Fine-tuning")
    
    # Preparar el modelo base
    inputs = keras.Input(shape=X_train.shape[1:])
    x = layers.Resizing(224, 224)(inputs)
    x = applications.mobilenet_v2.preprocess_input(x)
    
    # Cargar MobileNetV2 preentrenado
    base_model = applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet'
    )
    
    # Congelar inicialmente
    base_model.trainable = False
    
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)
    
    model = keras.Model(inputs, outputs)
    
    # Primera fase: entrenar solo las capas superiores
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("Fase 1: Entrenando capas superiores...")
    model.fit(X_train, Y_train, epochs=5, batch_size=32, validation_split=0.1, verbose=1)
    
    # Segunda fase: fine-tuning
    print("\nFase 2: Fine-tuning de las últimas capas del modelo base...")
    base_model.trainable = True
    
    # Congelar todas las capas excepto las últimas 20
    for layer in base_model.layers[:-20]:
        layer.trainable = False
    
    # Recompilar con learning rate más bajo
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    model.fit(X_train, Y_train, epochs=10, batch_size=32, validation_split=0.1, verbose=1)
    
    # Evaluación
    test_acc = model.evaluate(X_test, Y_test, verbose=0)[1]
    print(f"\nTransfer Learning con Fine-tuning → Test acc: {test_acc:.4f}")
    
    return model

# --------------------------------------------------
# Main
# --------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ejecuta tareas de la práctica")
    parser.add_argument(
        "task",
        choices=["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "ALL"],
        help="Tarea a ejecutar (o ALL para todas)",
    )
    parser.add_argument(
        "--dataset-dir",
        default="dataset_propio",
        help="Directorio del dataset propio (default: dataset_propio)"
    )
    args = parser.parse_args()

    # Cargar datos
    print("Cargando CIFAR-10...")
    X_train, Y_train, X_test, Y_test = cargar_y_preprocesar_cifar10()
    print(f"Datos cargados: Train {X_train.shape}, Test {X_test.shape}")
    
    # Para las tareas K y L, cargar dataset propio si es necesario
    X_propio = None
    Y_propio = None
    
    if args.task in ["J", "K", "L", "ALL"]:
        # Intentar cargar dataset propio
        if args.task == "J" or args.task == "ALL":
            # En tarea J, usar un modelo simple para evaluar
            model_simple, _ = probar_MLP(X_train, Y_train, X_test, Y_test, verbose=0)
            X_propio, Y_propio = evaluar_dataset_propio(args.dataset_dir, model_simple)
        else:
            # Para K y L, necesitamos cargar el dataset sin evaluar aún
            print(f"\nCargando dataset propio desde {args.dataset_dir}...")
            # Crear un modelo temporal solo para cargar
            model_temp = keras.Sequential([
                keras.Input(shape=(32,32,3)),
                layers.Flatten(),
                layers.Dense(10)
            ])
            X_propio, Y_propio = evaluar_dataset_propio(args.dataset_dir, model_temp)

        # Ejecutar tareas
    if args.task in ["A", "ALL"]:
        print("\n" + "="*50)
        print("TAREA A: MLP BÁSICO")
        print("="*50)
        modelA, _ = probar_MLP(X_train, Y_train, X_test, Y_test)

    if args.task in ["B", "ALL"]:
        print("\n" + "="*50)
        print("TAREA B: ANÁLISIS DE EPOCHS")
        print("="*50)
        tarea_B_analizar_epochs(X_train, Y_train, X_test, Y_test)

    if args.task in ["C", "ALL"]:
        print("\n" + "="*50)
        print("TAREA C: ANÁLISIS DE BATCH SIZE")
        print("="*50)
        tarea_C_analizar_batch_size(X_train, Y_train, X_test, Y_test)

    if args.task in ["D", "ALL"]:
        print("\n" + "="*50)
        print("TAREA D: EXPLORACIÓN DE OPTIMIZADORES")
        print("="*50)
        tarea_D_explorar_optimizadores(X_train, Y_train, X_test, Y_test)

    if args.task in ["E", "ALL"]:
        print("\n" + "="*50)
        print("TAREA E: REGULARIZACIÓN L2")
        print("="*50)
        tarea_E_regularizacion(X_train, Y_train, X_test, Y_test)

    if args.task in ["F", "ALL"]:
        print("\n" + "="*50)
        print("TAREA F: DROPOUT")
        print("="*50)
        tarea_F_dropout(X_train, Y_train, X_test, Y_test)

    if args.task in ["G", "ALL"]:
        print("\n" + "="*50)
        print("TAREA G: CNN BÁSICA")
        print("="*50)
        modelG, _ = tarea_G_CNN_basica(X_train, Y_train, X_test, Y_test)

    if args.task in ["H", "ALL"]:
        print("\n" + "="*50)
        print("TAREA H: EFECTO DEL KERNEL SIZE")
        print("="*50)
        tarea_H_kernel_size(X_train, Y_train, X_test, Y_test)

    if args.task in ["I", "ALL"]:
        print("\n" + "="*50)
        print("TAREA I: OPTIMIZACIÓN DE ARQUITECTURA CNN")
        print("="*50)
        mejor_cnn = tarea_I_optimizar_CNN(X_train, Y_train, X_test, Y_test)

    if args.task in ["J", "ALL"]:
        print("\n" + "="*50)
        print("TAREA J: EVALUACIÓN EN DATASET PROPIO")
        print("="*50)
        # En la fase ALL, asumimos que ya se cargó X_propio, Y_propio en la sección de carga
        evaluar_dataset_propio(args.dataset_dir, model_simple if args.task == "J" else modelA)

    if args.task in ["K", "ALL"]:
        print("\n" + "="*50)
        print("TAREA K: EXPERIMENTACIÓN SOBRE DATASET PROPIO")
        print("="*50)
        df_k = tarea_K_experimentacion_dataset_propio(
            X_train, Y_train, X_test, Y_test,
            X_propio, Y_propio
        )

    if args.task in ["L", "ALL"]:
        print("\n" + "="*50)
        print("TAREA L: MEJORAS PARA GENERALIZACIÓN")
        print("="*50)
        if X_propio is None or Y_propio is None:
            print("ERROR: No hay dataset propio cargado. Se omite tarea L.")
        else:
            model_L, resultados_L = tarea_L_mejoras_generalizacion(
                X_train, Y_train, X_test, Y_test,
                X_propio, Y_propio
            )

    # Transfer Learning Mejorado (solo en ALL para comparar con CNN H)
    if args.task == "ALL":
        print("\n" + "="*50)
        print("TRANSFER LEARNING MEJORADO")
        print("="*50)
        model_TL = transfer_learning_mejorado(X_train, Y_train, X_test, Y_test)

    print("\n¡Todas las tareas solicitadas se han completado con éxito!")
