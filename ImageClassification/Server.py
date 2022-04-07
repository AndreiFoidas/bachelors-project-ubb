
import os
from werkzeug.utils import secure_filename
from flask import Flask, json, request, jsonify

from ImageClassification import ImageClassification
from OCR import OCR

imageClassification: ImageClassification = None
ocr: OCR = None


UPLOAD_FOLDER = "ServerPhotos"
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

api = Flask(__name__)
api.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

test = [{"id": 1, "name": "andrei"}]


@api.route('/test', methods=['GET'])
def get_test():
    return json.dumps(test)


@api.route('/uploadPhoto', methods=['POST'])
def post_upload_photo():
    if request.files.get("uploaded_file") is None:
        return "fail"

    print(request.files)
    file = request.files["uploaded_file"]
    file.save(os.path.join(api.config['UPLOAD_FOLDER'], secure_filename(file.filename)))

    guessedPlastic, indexMax, maxValue = Classify_Photo(UPLOAD_FOLDER+"/"+file.filename)

    return jsonify(
        nr=indexMax,
        name=guessedPlastic,
        percentage=maxValue,
        result="success",
    )

def Classify_Photo(filePath):
    global imageClassification, ocr
    plastics = ["1_PET", "2_HDPE", "3_PVC", "4_LDPE", "5_PP", "6_PS", "7_OTHER", "8_NOPLASTIC"]
    print(filePath)
    pred = [0.0] * 8
    tempPred = [0.0] * 8

    tempPred = imageClassification.Classify_Photo_VGG19(filePath, False)
    print("Vgg19: ")
    print(tempPred)
    pred = [a + b for a, b in zip(pred, tempPred)]

    tempPred = imageClassification.Classify_Photo_EfficientNet(filePath, False)
    print("EffNet: ")
    print(tempPred)
    pred = [a + b for a, b in zip(pred, tempPred)]

    #tempPred = ocr.Classify_Photo_OCR(filePath, True)
    tempPred = [0.0] * 8
    print("OCR: ")
    print(tempPred)
    pred = [a + b for a, b in zip(pred, tempPred)]

    print("Final result: ")
    predDecimal = [0] * 8
    for i in range(len(pred)):
        predDecimal[i] = ("%.17f" % pred[i]).rstrip('0').rstrip('.')
    print(pred)
    print(predDecimal)
    maxValue = max(predDecimal)
    indexMax = predDecimal.index(maxValue)
    guessedPlastic = plastics[indexMax]

    print("\nResult:")
    print(maxValue)
    print(guessedPlastic)

    return guessedPlastic, indexMax, maxValue

def Start_Server():
    global imageClassification, ocr
    imageClassification = ImageClassification()
    ocr = OCR()
    imageClassification.Load_Models()

    # api.run("192.168.0.148", port=420)
    api.run()


if __name__ == '__main__':
    Start_Server()
