
from google.cloud import vision
import io
import os

class OCR:
    def __init__(self):
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "files/ocr-recycled-plastic.json"

    def Detect_Text_In_Local_File(self, path, printOutput):
        client = vision.ImageAnnotatorClient()

        with io.open(path, 'rb') as image_file:
            content = image_file.read()

        image = vision.Image(content=content)

        response = client.text_detection(image=image)
        texts = response.text_annotations
        if printOutput:
            print('Texts:')

        for text in texts:
            if printOutput:
                print('\n"{}"'.format(text.description))

            vertices = (['({},{})'.format(vertex.x, vertex.y)
                         for vertex in text.bounding_poly.vertices])

            if printOutput:
                print('bounds: {}'.format(','.join(vertices)))

        if response.error.message:
            raise Exception(
                '{}\nFor more info on error messages, check: '
                'https://cloud.google.com/apis/design/errors'.format(
                    response.error.message))

        return texts

    def Similarity_Percentage(self, text, label_array, number_array):
        similarity_sum = 0

        for label in label_array:
            if text.casefold() == label.casefold():
                similarity_sum += 0.66

        for number in number_array:
            if text == number:
                similarity_sum += 0.33

        return similarity_sum

    def Classify_Photo_OCR(self, photo_path, printText):
        pred = [0.0] * 8

        info = self.Detect_Text_In_Local_File(photo_path, False)

        if len(info) == 0:
            pred[7] += 0.99
            return pred

        item = info[0]

        texts = item.description
        if printText:
            print(texts)
        texts = texts.split()

        for text in texts:
            # 1 PET
            pred[0] += self.Similarity_Percentage(text, ["PET", "PETE"], ["1", "01"])
            # 2 HDPE
            pred[1] += self.Similarity_Percentage(text, ["HDPE", "PEHD", "HD-PE", "PE-HD", "PE"], ["2", "02"])
            # 3 PVC
            pred[2] += self.Similarity_Percentage(text, ["PVC"], ["3", "03"])
            # 4 LDPE
            pred[3] += self.Similarity_Percentage(text, ["LDPE", "PELD", "PE-LD", "LD-PE"], ["4", "04"])
            # 5 PP
            pred[4] += self.Similarity_Percentage(text, ["PP"], ["5", "05"])
            # 6 PS
            pred[5] += self.Similarity_Percentage(text, ["PS"], ["6", "06"])
            # 7 OTHER
            pred[6] += self.Similarity_Percentage(text, ["OTHER", "O"], ["7", "07", "0"])
            # 8 NOT PLASTIC
            pred[7] += self.Similarity_Percentage(text, ["", "PAP", "GL", "e", "ce", "BIO", "VEGAN", "FSC", "6M", "12M", "24M", "ALU", "FE"],
                                                  ["21", "90", "70", "84", "14", "81", "40"])

        for i in range(len(pred)):
            if pred[i] > 1:
                pred[i] = 1

        return pred


if __name__ == '__main__':
    ocr = OCR()

