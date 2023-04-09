# для запуска
# python real_time_object_detection.py --prototxt MobileNetSSD_deploy.prototxt.txt --model MobileNetSSD_deploy.caffemodel

# подключение пакетов
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2

# создание аргумента и настройка
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True,
	help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
	help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.2,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# Иницилизация классов
# создание цветов для границ обьектов
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# загрузка нашей сериализованной модели с диска
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# Инилазцтя камеры или записи
# и фпс счетчика
print("[INFO] starting video stream...")
# vs = VideoStream(src=0).start()
vs = cv2.VideoCapture('D:\поезд.mp4')
time.sleep(2.0)
fps = FPS().start()

# цикл для каждого фрейма видео
while True:
	# возьмите кадр из потокового видеопотока и измените его размер

	ret,frame = vs.read()
	if not ret:
		break
	# так, чтобы максимальная ширина составляла 400 пикселей
	# frame = imutils.resize(frame, width=400)

	# возьмите размеры рамки и преобразуйте ее в большой объект
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
		0.007843, (300, 300), 127.5)

	# пропустите большой двоичный объект по сети и получите данные обнаружения и прогнозы

	net.setInput(blob)
	detections = net.forward()

	# зацикливаться на обнаружениях
	for i in np.arange(0, detections.shape[2]):
		# извлеките достоверность (т.е. вероятность), связанную с прогнозом
		confidence = detections[0, 0, i, 2]

		# отфильтруйте слабые обнаружения, убедившись, что `достоверность`
		# превышает минимальную достоверность
		if confidence > args["confidence"]:
			# извлеките индекс метки класса из
			#`обнаружения`, затем вычислите (x, y)-координаты
			#ограничивающая рамка для объекта
			idx = int(detections[0, 0, i, 1])
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# нарисуйте предсказание на рамке
			label = "{}: {:.2f}%".format(CLASSES[idx],
				confidence * 100)
			cv2.rectangle(frame, (startX, startY), (endX, endY),
				COLORS[idx], 2)
			y = startY - 15 if startY - 15 > 15 else startY + 15
			cv2.putText(frame, label, (startX, y),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

	# показать выходной кадр
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# если была нажата клавиша "q", прервите цикл
	if key == ord("q"):
		break

	# обновите счетчик кадров в секунду
	fps.update()

# остановите таймер и отобразите информацию о частоте кадров в секунду
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
















