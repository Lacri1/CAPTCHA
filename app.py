from flask import Flask, request, render_template
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import load_model
from layer import CTCLayer

app = Flask(__name__)

# 모델 로드
model = tf.keras.models.load_model('captcha.h5', custom_objects={'CTCLayer': CTCLayer}, compile=False)

# 모델을 생성할 때 사용한 규칙 적용
char_to_num = tf.keras.layers.StringLookup(vocabulary=['2', '3', '4', '5', '6', '7', '8', 'b', 'c', 'd', 'e', 'f', 'g', 'm', 'n', 'p', 'w', 'x', 'y'], mask_token=None)
num_to_char = tf.keras.layers.StringLookup(vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True)
img_width = 200
img_height = 50


# 캡차 예측
def predict_captcha(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_png(img, channels=1)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, [img_height, img_width])
    img = tf.transpose(img, perm=[1, 0, 2])
    img = tf.expand_dims(img, axis=0)

    prediction = model.predict([img, tf.convert_to_tensor([[[1.0]]])])
    decoded_indices = tf.argmax(prediction, axis=2)[0]
    decoded_text = ''.join(
        [char_to_num.get_vocabulary()[i] for i in decoded_indices if i < len(char_to_num.get_vocabulary())])
    return decoded_text


# 중복 제거
def remove_duplicates(input_string):
    result = ''
    for char in input_string:
        if char not in result:
            result += char
    return result


# 웹 페이지 라우트
@app.route('/', methods=['GET', 'POST'])
def index():
    filename = None
    cleaned_result = None
    if request.method == 'POST':
        # 업로드 된 이미지 파일을 받아옴
        uploaded_file = request.files['file']
        if uploaded_file.filename != '':
            image_path = "temp_image.png"  # 임시 이미지 파일 경로
            uploaded_file.save(image_path)
            captcha_result = predict_captcha(image_path)
            cleaned_result = remove_duplicates(captcha_result)  # 중복 문자 제거

    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
