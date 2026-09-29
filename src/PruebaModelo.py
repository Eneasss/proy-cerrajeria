import os
import json
import time
import cv2
from matplotlib.pyplot import gray
import numpy as np

# Soporte para tflite-runtime (PC de mostrador) o tensorflow completo
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow.lite as tflite

# ==========================================
# CONFIGURACIÓN Y PARÁMETROS
# ==========================================
MODEL_PATH = os.path.join("model", "key_classifier.tflite")
LABELS_PATH = os.path.join("model", "labels.json")
CONFIDENCE_THRESHOLD = 0.70  # 70% mínimo para considerar predicción válida
CAMERA_INDEX = 0             # 0 para cámara principal / integrada, 1 para USB externa
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720


def load_resources():
    """Valida y carga el modelo TFLite y las etiquetas."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"No se encontró el modelo en: {MODEL_PATH}")
    if not os.path.exists(LABELS_PATH):
        raise FileNotFoundError(f"No se encontró el archivo de etiquetas en: {LABELS_PATH}")

    with open(LABELS_PATH, "r") as f:
        labels = json.load(f)

    interpreter = tflite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    _, target_h, target_w, _ = input_details[0]["shape"]

    return interpreter, input_details, output_details, labels, (target_w, target_h)


def main():
    try:
        interpreter, input_details, output_details, labels, (target_w, target_h) = load_resources()
    except Exception as e:
        print(f"[ERROR] {e}")
        return

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not cap.isOpened():
        print(f"[ERROR] No se pudo abrir la cámara con índice {CAMERA_INDEX}")
        return

    print("Cámara iniciada. Presioná 'Q' o 'ESC' para salir.")

    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Falló la lectura del cuadro.")
            break

        h, w, _ = frame.shape

        # ==========================================
        # 1. REGIÓN DE INTERÉS DINÁMICA (AUTO-ENCUADRE)
        # ==========================================

        # Parámetros calibrados para la escala de la llave
        MIN_KEY_AREA = 2500    # Llaves chicas (ej. cilindro)
        MAX_KEY_AREA = 35000   # Si supera esto, es una sombra, la hoja entera o la mesa
        BORDER_MARGIN = 15     # Píxeles de margen con los bordes de la cámara

        # Aislamiento de la llave sobre el fondo blanco
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)

        # Otsu inverso: la llave (oscura) pasa a blanco y el papel pasa a negro
        # 1. Umbral adaptativo: inmune a gradientes de iluminación y sombras suaves
        thresh = cv2.adaptiveThreshold(
            blurred, 
            255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 
            blockSize=51, 
            C=12
        )

        # 2. Limpieza morfológica: rellena reflejos internos del metal y borra motas
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


        valid_candidates = []
        for c in contours:
            area = cv2.contourArea(c)
            # Filtrar solo si tiene el área de una llave
            if MIN_KEY_AREA < area < MAX_KEY_AREA:
                kx, ky, kw, kh = cv2.boundingRect(c)
                # Descartar si el contorno toca los bordes exteriores de la cámara
                if (kx > BORDER_MARGIN and ky > BORDER_MARGIN and 
                    (kx + kw) < (w - BORDER_MARGIN) and 
                    (ky + kh) < (h - BORDER_MARGIN)):
                    valid_candidates.append(c)

        """ # Filtrar contornos grandes para descartar sombras suaves o motas de polvo
        valid_contours = [c for c in contours if cv2.contourArea(c) > 3000] """

        if valid_candidates:
            # Seleccionar el contorno principal (la llave)
            key_cnt = max(valid_candidates, key=cv2.contourArea)
            kx, ky, kw, kh = cv2.boundingRect(key_cnt)

            # El lado del cuadrado se define por la dimensión mayor (normalmente el largo de la llave)
            # con un 15% de margen extra (padding)
            max_side = max(kw, kh)
            box_size = int(max_side * 1.25)
            box_size = min(box_size, min(h, w)-20)  # No exceder los límites de la cámara

            # Calcular centro de la llave detectada
            center_x = kx + kw // 2
            center_y = ky + kh // 2

            # Construir recorte cuadrado centrado en la llave
            x1 = max(0, min(w - box_size, center_x - box_size // 2))
            y1 = max(0, min(h - box_size, center_y - box_size // 2))
            x2 = x1 + box_size
            y2 = y1 + box_size

            roi = frame[y1:y2, x1:x2]
        else:
            # Recuadro por defecto si no hay ninguna llave en la mesa
            box_size = int(min(h, w) * 0.42)
            x1 = (w - box_size) // 2
            y1 = (h - box_size) // 2
            x2 = x1 + box_size
            y2 = y1 + box_size
            roi = frame[y1:y2, x1:x2]

        # ==========================================
        # 2. PREPROCESAMIENTO E INFERENCIA TFLITE
        # ==========================================
        # Conversión BGR a RGB y redimensionado al input del modelo (224x224)
        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        roi_resized = cv2.resize(roi_rgb, (target_w, target_h))
        input_tensor = np.expand_dims(roi_resized, axis=0).astype(np.float32)

        # Inferencia
        start_inf = time.time()
        interpreter.set_tensor(input_details[0]["index"], input_tensor)
        interpreter.invoke()
        output_probs = interpreter.get_tensor(output_details[0]["index"])[0]
        inf_latency_ms = (time.time() - start_inf) * 1000

        # Interpretación
        top_idx = int(np.argmax(output_probs))
        confidence = float(output_probs[top_idx])
        detected_label = labels[top_idx]
        is_confident = confidence >= CONFIDENCE_THRESHOLD

        # ==========================================
        # 3. INTERFAZ VISUAL EN PANTALLA
        # ==========================================
        # Verde si supera el umbral, Naranja/Ámbar si hay duda
        box_color = (0, 255, 0) if is_confident else (0, 165, 255)

        # Dibujar recuadro guía para la llave
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

        # Banner de resultado sobre el cuadro
        status_text = f"{detected_label}: {confidence * 100:.1f}%" if is_confident else f"Incierto ({detected_label}: {confidence * 100:.1f}%)"
        
        # Fondo oscuro para legibilidad del texto
        text_size, _ = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.rectangle(frame, (x1, y1 - 35), (x1 + text_size[0] + 16, y1), (20, 20, 20), -1)
        cv2.putText(frame, status_text, (x1 + 8, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, box_color, 2)

        # Cálculo de FPS de visualización
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
        prev_time = curr_time

        # Métricas de rendimiento en esquina superior
        cv2.putText(frame, f"FPS: {fps:.1f} | Latencia CPU: {inf_latency_ms:.1f} ms", 
                    (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 220, 220), 2)
        cv2.putText(frame, "Centra la llave dentro del recuadro", 
                    (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)

        cv2.imshow("Detector de Llaves en Tiempo Real - TFLite", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q'), 27):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()