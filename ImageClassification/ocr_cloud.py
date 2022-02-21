
from google.cloud import vision
import io
import os

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "ocr-recycled-plastic.json"


def detect_text(path):
    """Detects text in the file."""

    client = vision.ImageAnnotatorClient()

    with io.open(path, 'rb') as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    response = client.text_detection(image=image)
    texts = response.text_annotations
    print('Texts:')

    for text in texts:
        print('\n"{}"'.format(text.description))

        vertices = (['({},{})'.format(vertex.x, vertex.y)
                     for vertex in text.bounding_poly.vertices])

        print('bounds: {}'.format(','.join(vertices)))

    if response.error.message:
        raise Exception(
            '{}\nFor more info on error messages, check: '
            'https://cloud.google.com/apis/design/errors'.format(
                response.error.message))


def main():
    res_path = "testPhotos"

    for img in os.listdir(res_path):
        img_name = img
        image_path = res_path + "/" + img
        # img_arr = cv2.imread(image_path)

        print("Testing " + str(image_path))

        detect_text(image_path)


main()
# detect_text("testPhotos/is1.jpg")
