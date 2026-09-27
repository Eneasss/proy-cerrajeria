import os
import glob
import cv2

# ==========================================
# CONFIGURACIÓN GENERAL
# ==========================================
DATASET_DIR = "dataset"  # Carpeta raíz compatible con image_dataset_from_directory
CAMERA_INDEX = 1         # 0 para cámara principal, 1 o 2 para webcam USB externa
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720


def get_next_index(target_dir, label):
    """Calcula el siguiente índice disponible para no sobreescribir imágenes existentes."""
    files = glob.glob(os.path.join(target_dir, f"{label}_*.jpg"))
    if not files:
        return 0
    indices = []
    for f in files:
        filename = os.path.basename(f)
        try:
            # Extrae el número del patrón 'nombre_0042.jpg'
            num_part = filename.replace(f"{label}_", "").replace(".jpg", "")
            indices.append(int(num_part))
        except ValueError:
            continue
    return max(indices) + 1 if indices else len(files)


def main():
    print("=" * 50)
    print(" GESTOR DE CAPTURA DE MUESTRAS - CERRAJERÍA")
    print("=" * 50)

    # 1. Definir etiqueta/modelo inicial
    label_name = input("Ingrese el nombre del modelo de llave (ej: trabex_tbx1, yale_y1): ").strip().lower()
    while not label_name:
        label_name = input("El nombre no puede estar vacío. Ingrese el modelo: ").strip().lower()

    save_dir = os.path.join(DATASET_DIR, label_name)
    os.makedirs(save_dir, exist_ok=True)
    count = get_next_index(save_dir, label_name)

    # 2. Inicializar cámara
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not cap.isOpened():
        print(f"\n[ERROR] No se pudo conectar a la cámara con índice {CAMERA_INDEX}.")
        return

    print("\nControles de la ventana:")
    print("  • [ESPACIO] : Tomar fotografía y guardar")
    print("  • [C]       : Cambiar de modelo sin reiniciar el script")
    print("  • [Q / ESC] : Cerrar el programa\n")

    flash_frames = 0  # Indicador visual al disparar

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Fallo en la lectura del cuadro de video.")
            break

        h, w, _ = frame.shape
        display = frame.copy()

        # Coordenadas del cuadro delimitador guía (ROI centrado)
        box_w, box_h = int(w * 0.5), int(h * 0.6)
        x1 = (w - box_w) // 2
        y1 = (h - box_h) // 2
        x2 = x1 + box_w
        y2 = y1 + box_h

        # Dibujar recuadro guía y textos informativos
        color_box = (0, 255, 0) if flash_frames == 0 else (255, 255, 255)
        cv2.rectangle(display, (x1, y1), (x2, y2), color_box, 2)

        # Panel de estado en pantalla
        cv2.putText(display, f"Modelo: {label_name}", (30, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        cv2.putText(display, f"Muestras: {count}", (30, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.putText(display, "[ESPACIO]: Capturar | [C]: Cambiar | [Q]: Salir", (30, h - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        if flash_frames > 0:
            flash_frames -= 1

        cv2.imshow("Captura de Muestras de Llaves", display)
        key = cv2.waitKey(1) & 0xFF

        # --- EVENTOS DE TECLADO ---
        if key == 32:  # BARRA ESPACIADORA
            filename = f"{label_name}_{count:04d}.jpg"
            file_path = os.path.join(save_dir, filename)

            # Se guarda el fotograma completo original (sin los textos ni el recuadro verde)
            cv2.imwrite(file_path, frame)
            print(f"[OK] Guardada: {file_path}")

            count += 1
            flash_frames = 3  # Efecto visual de obturador

        elif key in (ord('c'), ord('C')):  # CAMBIAR DE MODELO
            print("\n" + "-" * 40)
            nuevo_label = input("Ingrese el nuevo modelo de llave: ").strip().lower()
            if nuevo_label:
                label_name = nuevo_label
                save_dir = os.path.join(DATASET_DIR, label_name)
                os.makedirs(save_dir, exist_ok=True)
                count = get_next_index(save_dir, label_name)
                print(f"[CAMBIO] Directorio activo: {save_dir} (comenzando en índice {count})")
            print("-" * 40 + "\n")

        elif key in (ord('q'), ord('Q'), 27):  # SALIR
            print("\nFinalizando captura...")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()