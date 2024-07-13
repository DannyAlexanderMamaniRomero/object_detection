from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
import torch

app = Flask(__name__)

# Cargar el modelo YOLOv5
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # Puedes cambiar 'yolov5s' a 'yolov5m', 'yolov5l', etc. para diferentes versiones del modelo.

# Inicializar la captura de video
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("Error: Could not open video capture")
    exit()

# Establecer el tamaño del frame para la detección
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Definir una variable para almacenar el frame capturado
captured_frame = None

# Definir una variable para controlar si se ha detectado un objeto
object_detected = False

def generate():
    global captured_frame, object_detected
    while True:
        # Leer el frame más reciente de la cámara
        ret, frame = cap.read()
        if not ret:
            break  # Si no se puede leer el frame, salir del bucle

        # Reducción de la resolución de entrada para mejorar el rendimiento
        small_frame = cv2.resize(frame, (640, 480))

        # Convertir la imagen para el modelo
        img = small_frame[...,::-1]  # Convert BGR to RGB
        results = model(img)  # Ejecutar la detección

        # Procesar resultados de detección
        detections = results.pandas().xyxy[0]  # Convertir resultados a un DataFrame
        for _, row in detections.iterrows():
            x1, y1, x2, y2, conf, cls = row[:6]
            label = f'{model.names[int(cls)]} {conf:.2f}'
            color = (0, 255, 0)  # Color de la caja de detección
            cv2.rectangle(small_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(small_frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Captura del frame cuando se detecta un objeto
            if not object_detected and conf > 0.5:  # Ajusta el umbral de confianza si es necesario
                object_detected = True
                captured_frame = small_frame.copy()  # Guardar una copia del frame

        # Mostrar el frame capturado con el objeto detectado en una ventana separada
        if captured_frame is not None:
            cv2.imshow("Captured Frame", captured_frame)

        # Convertir la imagen de nuevo a BGR para OpenCV
        output_frame = small_frame

        # Convertir el frame a un formato JPEG para transmitirlo al navegador
        (flag, encodedImage) = cv2.imencode(".jpg", output_frame)
        if not flag:
            continue
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')

        # Terminar el programa si se presiona la tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video_feed")
def video_feed():
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/capture_object")
def capture_object():
    global object_detected, captured_frame
    if captured_frame is not None and object_detected:
        # Guarda la imagen capturada
        cv2.imwrite('captured_object.jpg', captured_frame)
        # Resetear variables
        captured_frame = None
        object_detected = False
        return jsonify({'message': '¡Objeto capturado y guardado como "captured_object.jpg"!'})

    return jsonify({'message': 'No se ha detectado ningún objeto.'})

if __name__ == "__main__":
    try:
        print("Starting Flask server...")
        app.run(debug=True, use_reloader=False)
    finally:
        cap.release()
        print("Released video capture")