
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

@api.route('/uploadPhoto', methods=['POST'])
def post_upload_photo():
    if request.files.get("uploaded_file") is None:
        return "fail"

    print(request.files)
    file = request.files["uploaded_file"]
    file.save(os.path.join(api.config['UPLOAD_FOLDER'], secure_filename(file.filename)))

    guessedPlastic, indexMax, maxValue = Classify_Photo(UPLOAD_FOLDER+"/"+file.filename, True)

    return jsonify(
        nr=indexMax,
        name=guessedPlastic,
        percentage=maxValue,
        result="success",
        filename=file.filename,
    )

@api.route('/uploadInfo', methods=['POST'])
def post_upload_info():
    plastic = request.values["plastic"]
    filename = request.values["filename"]

    new_filename = "G" + plastic[0] + "-D" + filename
    print(new_filename)

    os.rename(UPLOAD_FOLDER + "/" + filename, UPLOAD_FOLDER + "/" + new_filename)

    return jsonify(
        status="success",
    )


def maxelements(seq):
    max_indices = []
    max_val = seq[0]
    for i, val in ((i, val) for i, val in enumerate(seq) if val >= max_val):
        if val == max_val:
            max_indices.append(i + 1)
        else:
            max_val = val
            max_indices = [i + 1]

    return max_indices

def Classify_Photo(filePath, printText):
    global imageClassification, ocr
    plastics = ["1 PET", "2 HDPE", "3 PVC", "4 LDPE", "5 PP", "6 PS", "7 OTHER", "8 NOT PLASTIC"]
    print(filePath)
    pred = [0.0] * 8
    tempPred = [0.0] * 8

    tempPred = imageClassification.Classify_Photo_VGG19(filePath, False)
    if printText:
        print(str(tempPred))
    predVgg19 = tempPred[:]

    tempPred = imageClassification.Classify_Photo_EfficientNet(filePath, False)
    if printText:
        print(str(tempPred))
    predEffnet = tempPred[:]

    tempPred = ocr.Classify_Photo_OCR(filePath, True)
    if printText:
        print(str(tempPred))
    #tempPred = [0.0] * 8
    predOCR = tempPred[:]

    # weighted version of the sum rule-based fusion method
    for i in range(len(pred)):
        pred[i] = 61 * predOCR[i] + 25 * predVgg19[i] + 14 * predEffnet[i]

    predDecimal = [0] * 8
    for i in range(len(pred)):
        predDecimal[i] = ("%.17f" % pred[i]).rstrip('0').rstrip('.')

    maxValue = max(predDecimal)
    indexMax = maxelements(predDecimal)
    print(indexMax)
    guessedPlastic = plastics[indexMax[0] - 1]

    print("\nResult:")
    print(maxValue)
    print(guessedPlastic)

    return guessedPlastic, indexMax[0] - 1, maxValue

def Start_Server():
    global imageClassification, ocr
    imageClassification = ImageClassification()
    ocr = OCR()
    imageClassification.Load_Models()

    api.run("192.168.0.148", port=420)
    # api.run()


if __name__ == '__main__':
    Start_Server()
