
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

    guessedPlastic, indexMax, maxValue = Classify_Photo(UPLOAD_FOLDER+"/"+file.filename)

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

def Classify_Photo(filePath):
    global imageClassification, ocr
    plastics = ["1 PET", "2 HDPE", "3 PVC", "4 LDPE", "5 PP", "6 PS", "7 OTHER", "8 NOT PLASTIC"]
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

    tempPred = ocr.Classify_Photo_OCR(filePath, True)
    #tempPred = [0.0] * 8
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

    api.run("192.168.0.148", port=420)
    # api.run()

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


def Classify_Photo_Test(filePath):
    global imageClassification, ocr
    file_object = open("test.txt", "a")
    plastics = ["1 PET", "2 HDPE", "3 PVC", "4 LDPE", "5 PP", "6 PS", "7 OTHER", "8 NOT PLASTIC"]
    print(filePath)
    pred = [0.0] * 8
    tempPred = [0.0] * 8

    file_object.write(filePath + "\n")

    tempPred = imageClassification.Classify_Photo_VGG19(filePath, False)
    file_object.write(str(tempPred) + str(maxelements(tempPred)) + "\n")
    predVgg19 = tempPred[:]

    tempPred = imageClassification.Classify_Photo_EfficientNet(filePath, False)
    file_object.write(str(tempPred) + str(maxelements(tempPred)) + "\n")
    predEffnet = tempPred[:]

    tempPred = ocr.Classify_Photo_OCR(filePath, True)
    #tempPred = [0.0] * 8
    file_object.write(str(tempPred) + str(maxelements(tempPred)) + "\n")
    predOCR = tempPred[:]


    for i in range(len(pred)):
        pred[i] = 61 * predOCR[i] + 25 * predVgg19[i] + 14 * predEffnet[i]


    predDecimal = [0] * 8

    for i in range(len(pred)):
        predDecimal[i] = ("%.17f" % pred[i]).rstrip('0').rstrip('.')
    file_object.write(str(predDecimal) + str(maxelements(predDecimal)) + "\n")
    maxValue = max(predDecimal)
    indexMax = predDecimal.index(maxValue)
    guessedPlastic = plastics[indexMax]
    file_object.write(str(guessedPlastic) + "\n\n")

    print(guessedPlastic)

    return maxelements(predDecimal)

def SupremeTest():
    global imageClassification, ocr
    imageClassification = ImageClassification()
    ocr = OCR()
    imageClassification.Load_Models()
    ctAll = 0
    ctCorrect = 0
    ctWrong = 0
    ctSemi = 0

    folder_name = "../TestingImages"
    for img in os.listdir(folder_name):
        img_name = folder_name + "/" + img
        print(img_name)
        maxx = Classify_Photo_Test(img_name)

        ctAll += 1
        if int(img[1]) in maxx:
            if len(maxx) == 1:
                ctCorrect += 1
            else:
                ctSemi += 1
        else:
            ctWrong += 1

    print("Guessed: " + str(ctCorrect) + ", got wrong: " + str(ctWrong) + ", was close: " + str(
        ctSemi) + "; out of " + str(ctAll))

if __name__ == '__main__':
    # Start_Server()
    SupremeTest()
