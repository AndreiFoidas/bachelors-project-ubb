
from google.cloud import vision
import io
import os

class OCR:
    def __init__(self):
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "files/ocr-recycled-plastic.json"
        self._test_path = "testPhotos"

    def Detect_Text_In_Local_File(self, path, printOutput):
        """Detects text in the file."""

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
                similarity_sum += 1

        for number in number_array:
            if text == number:
                similarity_sum += 0.5

        return similarity_sum

    def Classify_Photo_OCR(self, photo_path, printText):
        pred = [0.0] * 8

        info = self.Detect_Text_In_Local_File(photo_path, False)

        if len(info) == 0:
            pred[7] += 1.5
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

        return pred

    def Test_Photos(self):

        for img in os.listdir(self._test_path):
            image_path = self._test_path + "/" + img

            print("Testing " + str(image_path))

            # self.Detect_Text_In_Local_File(image_path)
            print(self.Classify_Photo_OCR(image_path, True))

    def maxelements(self, seq):
        max_indices = []
        max_val = seq[0]
        for i, val in ((i, val) for i, val in enumerate(seq) if val >= max_val):
            if val == max_val:
                max_indices.append(i + 1)
            else:
                max_val = val
                max_indices = [i + 1]

        return max_indices

    def Test_OCR_For_True_Accuracy(self):
        all_photos_test_path = "../TestingImages"
        ctAll = 0
        ctCorrect = 0
        ctWrong = 0
        ctSemi = 0

        print("\nName: " + "OCR" + "\n")
        for img in os.listdir(all_photos_test_path):
            img_name = img
            path = all_photos_test_path + "/" + img
            pred = self.Classify_Photo_OCR(path, True)
            maxx = self.maxelements(pred)

            # print("Guessed val: " + str(img_name) + " " + str(maxx))
            ctAll += 1
            if int(img_name[1]) in maxx:
                if len(maxx) == 1:
                    ctCorrect += 1
                else:
                    ctSemi += 1
            else:
                ctWrong += 1

        print("Guessed: " + str(ctCorrect) + ", got wrong: " + str(ctWrong) + ", was close: " + str(
            ctSemi) + "; out of " + str(ctAll))




if __name__ == '__main__':
    ocr = OCR()
    ocr.Test_OCR_For_True_Accuracy()
    # ocr.Test_Photos()
    # ocr.Classify_Photo_OCR("ServerPhotos/2022-03-23_13-14-47-754.jpg", True)
