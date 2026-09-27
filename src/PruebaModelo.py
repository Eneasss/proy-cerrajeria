import os
import json
import time
import cv2
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
        # 1. REGIÓN DE INTERÉS (ROI CUADRADA)
        # ==========================================
        # Cuadro centrado para enfocar la paleta/perfil sin deformar aspecto
        box_size = int(min(h, w) * 0.60)
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