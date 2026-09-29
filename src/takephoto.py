import os
import glob
import cv2

# ==========================================
# CONFIGURACIÓN GENERAL
# ==========================================
DATASET_DIR = "dataset"
CAMERA_INDEX = 0          # 0 para cámara principal o 1/2 según el puerto USB
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

# Tamaño del recorte cuadrado (ROI)
# 340 px asegura que una llave de 7-9 cm llene el ~85% del encuadre a 45 cm de altura
ROI_SIZE = 500


def get_existing_classes(dataset_dir):
    """Devuelve las carpetas de clases ya existentes en el dataset."""
    if not os.path.exists(dataset_dir):
        return []
    return sorted([d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))])


def select_class(dataset_dir):
    """Permite elegir una clase existente mediante un menú numérico para evitar errores de tipeo."""
    classes = get_existing_classes(dataset_dir)
    print("\n" + "=" * 50)
    print(" SELECCIÓN DE MODELO DE LLAVE")
    print("=" * 50)
    
    if classes:
        print("Modelos detectados en 'dataset/':")
        for idx, cls_name in enumerate(classes, 1):
            print(f"  [{idx}] {cls_name}")
        print(f"  [N] Crear una nueva carpeta / modelo")
        
        while True:
            opcion = input("\nSeleccione un número o 'N': ").strip()
            if opcion.upper() == 'N':
                break
            if opcion.isdigit() and 1 <= int(opcion) <= len(classes):
                return classes[int(opcion) - 1]
            print("[!] Opción inválida. Intente de nuevo.")

    # Si elige crear una nueva o la carpeta dataset está vacía
    while True:
        nombre = input("Ingrese el nombre exacto de la carpeta del modelo: ").strip().lower()
        if nombre:
            return nombre
        print("[!] El nombre no puede estar vacío.")


def get_next_index(target_dir, label):
    """Calcula el siguiente índice disponible para no sobreescribir imágenes existentes."""
    files = glob.glob(os.path.join(target_dir, f"{label}_*.jpg"))
    if not files:
        return 0
    indices = []
    for f in files:
        filename = os.path.basename(f)
        try:
            num_part = filename.replace(f"{label}_", "").replace(".jpg", "")
            indices.append(int(num_part))
        except ValueError:
            continue
    return max(indices) + 1 if indices else len(files)


def main():
    label_name = select_class(DATASET_DIR)
    save_dir = os.path.join(DATASET_DIR, label_name)
    os.makedirs(save_dir, exist_ok=True)
    count = get_next_index(save_dir, label_name)

    # Inicializar con DirectShow para evitar demoras en Windows
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not cap.isOpened():
        print(f"\n[ERROR] No se pudo conectar a la cámara (Índice: {CAMERA_INDEX}).")
        return

    print("\n" + "=" * 50)
    print(f" CAPTURANDO PARA: {label_name.upper()}")
    print("=" * 50)
    print("Controles:")
    print("  • [ESPACIO] : Guardar recorte cuadrado (ROI)")
    print("  • [C]       : Cambiar de modelo de llave")
    print("  • [Q / ESC] : Salir\n")

    flash_frames = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Fallo en la lectura del cuadro de video.")
            break

        h, w, _ = frame.shape
        display = frame.copy()

        # Coordenadas de la ROI cuadrada centrada
        x1 = (w - ROI_SIZE) // 2
        y1 = (h - ROI_SIZE) // 2
        x2 = x1 + ROI_SIZE
        y2 = y1 + ROI_SIZE

        # Cuadro guía (Verde normal, Blanco al disparar)
        box_color = (255, 255, 255) if flash_frames > 0 else (0, 255, 0)
        cv2.rectangle(display, (x1, y1), (x2, y2), box_color, 2)

        # Panel informativo en pantalla
        cv2.putText(display, f"Modelo: {label_name}", (30, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 255), 2)
        cv2.putText(display, f"Fotos guardadas: {count}", (30, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 0), 2)
        cv2.putText(display, "Centra la llave llenando el cuadro verde", (30, 125),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 2)
        cv2.putText(display, "[ESPACIO]: Capturar | [C]: Cambiar Modelo | [Q]: Salir", (30, h - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 220, 220), 2)

        if flash_frames > 0:
            flash_frames -= 1

        cv2.imshow("Captura de Muestras - Dataset", display)
        key = cv2.waitKey(1) & 0xFF

        # Captura
        if key == 32:  # Barra espaciadora
            # EXTRAER ÚNICAMENTE EL RECORTE CUADRADO
            roi = frame[y1:y2, x1:x2].copy()

            filename = f"{label_name}_{count:04d}.jpg"
            file_path = os.path.join(save_dir, filename)

            # Guardar con compresión de alta calidad (calidad JPEG 95%)
            cv2.imwrite(file_path, roi, [cv2.IMWRITE_JPEG_QUALITY, 95])
            print(f"[OK] Guardada en {label_name}: {filename} ({ROI_SIZE}x{ROI_SIZE} px)")

            count += 1
            flash_frames = 3

        # Cambiar de clase
        elif key in (ord('c'), ord('C')):
            label_name = select_class(DATASET_DIR)
            save_dir = os.path.join(DATASET_DIR, label_name)
            os.makedirs(save_dir, exist_ok=True)
            count = get_next_index(save_dir, label_name)
            print(f"\n[CAMBIO] Ahora guardando en: {label_name} (índice inicial: {count})")

        # Salir
        elif key in (ord('q'), ord('Q'), 27):
            print("\nFinalizando captura de imágenes...")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()