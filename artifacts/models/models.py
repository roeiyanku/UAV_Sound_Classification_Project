from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
import numpy as np
import tensorflow as tf


# =========================
# Classical anomaly models
# =========================

def get_ocsvm(nu=0.1, kernel='rbf', gamma='scale'):
    model = OneClassSVM(nu=nu, kernel=kernel, gamma=gamma)
    return model


def get_isolation_forest(n_estimators=100, contamination=0.1, random_state=42):
    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=random_state
    )
    return model


def train_model(model, x_train):
    model.fit(x_train)
    return model


def predict_model(model, x):
    return model.predict(x)


def get_anomaly_scores(model, x, model_name):
    if model_name == "ocsvm":
        return -model.decision_function(x).ravel()
    elif model_name == "isolation_forest":
        return -model.score_samples(x).ravel()
    else:
        return None


# =========================
# CNN model
# =========================

def build_cnn_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),

        tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((2, 2)),

        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((2, 2)),

        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((2, 2)),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


# =========================
# Teacher-Student model
# =========================

def build_teacher_student_models(input_dim):
    teacher = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu')
    ])

    student = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(128, activation='linear')
    ])

    teacher.trainable = False
    student.compile(optimizer='adam', loss='mse')

    return teacher, student


def train_teacher_student(teacher, student, x_train_normal, epochs=10, batch_size=32):
    teacher_targets = teacher.predict(x_train_normal, verbose=0)

    student.fit(
        x_train_normal,
        teacher_targets,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )
    return student


def predict_teacher_student(teacher, student, x_train_normal, x_test, percentile=95):
    train_teacher_out = teacher.predict(x_train_normal, verbose=0)
    train_student_out = student.predict(x_train_normal, verbose=0)
    train_errors = np.mean((train_teacher_out - train_student_out) ** 2, axis=1)

    threshold = np.percentile(train_errors, percentile)

    test_teacher_out = teacher.predict(x_test, verbose=0)
    test_student_out = student.predict(x_test, verbose=0)
    test_errors = np.mean((test_teacher_out - test_student_out) ** 2, axis=1)

    preds = (test_errors > threshold).astype(int)

    return preds, test_errors, threshold
