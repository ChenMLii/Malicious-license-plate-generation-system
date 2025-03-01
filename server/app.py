from flask import Flask, request, jsonify, send_from_directory, send_file
import os
import uuid
from urllib.parse import urljoin
from flask_cors import CORS
from process_one_pic import AdvGan_process_image, CW_process_image,FGSM_process_image, PGD_process_image
from test_one_picture import getlb

app = Flask(__name__)
CORS(app, supports_credentials=True,resources=r'/*')
######################
# 配置文件
######################
FILEPATH =''

UPLOAD_FOLDER = 'static/uploads'
if not os.path.isdir(UPLOAD_FOLDER):
    os.mkdir(UPLOAD_FOLDER)
ADVIMG_FOLDER = 'static/advimg'
if not os.path.isdir(ADVIMG_FOLDER):
    os.mkdir(ADVIMG_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ADVIMG_FOLDER'] = ADVIMG_FOLDER


# 允许的扩展名
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

# 1M
app.config['MAX_CONTENT_LENGTH'] = 1 * 1024 * 1024

# 检查后缀名是否为允许的文件
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# 获取文件名
def get_filename1(filename):
    ext = os.path.splitext(filename)[-1]
    uuidOne = uuid.uuid4()
    prefix="gan"
    return prefix+"_"+str(uuidOne) + ext

def get_filename(filename,src):
    ext = os.path.splitext(filename)[-1]
    uuidOne = uuid.uuid4()
    if src=="0":
        prefix="source"
    else:
        prefix="target"
    return prefix+"_"+str(uuidOne) + ext
# 上传文件


@app.route("/upload", methods=['POST'])
def upload():
    filePath = UPLOAD_FOLDER
    pairs = os.listdir(filePath)
    src = request.args.get("src")
    src_pair=""
    tar_pair= ""
    if src == "0":
        for pair in pairs:
            if "source" in pair:
                src_pair = "./static/uploads/" + pair
                break
        if src_pair!="":
            os.remove(src_pair)
    else:
        for pair in pairs:
            if "target" in pair:
                tar_pair = "./static/uploads/" + pair
                break
        if tar_pair!="":
            os.remove(tar_pair)
    file = request.files.get('file')
    if file and allowed_file(file.filename):
        print(file.filename)
        filename = file.filename
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        global FILEPATH
        FILEPATH = filepath
        file.save(os.path.join(app.root_path, filepath))
        file_url = urljoin(request.host_url, filepath)
        return file_url
    return "not allow ext"

@app.route("/uploadGAN", methods=['POST'])
def uploadGAN():
    filePath = './static/uploadsGAN'
    pairs = os.listdir(filePath)
    src_pair=''
    for pair in pairs:
        if "gan" in pair:
            src_pair = "./static/uploadsGAN/" + pair
            break
    if src_pair != "":
        os.remove(src_pair)
    file = request.files.get('file')
    if file and allowed_file(file.filename):
        print(file.filename)
        filename = file.filename
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        global FILEPATH
        FILEPATH = filepath
        file.save(os.path.join(app.root_path, filepath))
        file_url = urljoin(request.host_url, filepath)
        return file_url
    return "not allow ext"
# 获取文件
@app.route('/uploads/<path:filename>')
def get_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
@app.route('/uploadsGAN/<path:filename>')
def get_file1(filename):
    return send_from_directory('./uploadsGAN', filename)

@app.route('/advimg/<path:filename>')
def get_advimg(filename):
    return send_from_directory(app.config['ADVIMG_FOLDER'], filename)

global NEW_FILEPATH
# 处理adv图片
@app.route('/process_image', methods=['GET'])
def process_image():
    global FILEPATH
    global NEW_FILEPATH
    print(FILEPATH)
    attack = request.args.get('attack')
    new_filepath=''
    if attack == 'AdvGan':
        new_filepath = AdvGan_process_image(FILEPATH)
        NEW_FILEPATH = new_filepath
    elif attack == 'FGSM':
        eps = request.args.get('eps')
        new_filepath = FGSM_process_image(FILEPATH, eps)
        NEW_FILEPATH = new_filepath
    elif attack == 'PGD':
        eps = request.args.get('eps')
        iters = request.args.get('iters')
        new_filepath = PGD_process_image(FILEPATH, eps, iters)
        NEW_FILEPATH = new_filepath
    elif attack == 'CandW':
        eps = request.args.get('eps')
        iters = request.args.get('iters')
        confidence = request.args.get('confidence')
        c = request.args.get('c')
        new_filepath = CW_process_image(FILEPATH, iters, confidence, c)
        NEW_FILEPATH = new_filepath

    
    # 获取图片文件的 URL
    image_url = request.host_url + new_filepath
    return jsonify({'imageUrl': image_url})

@app.route('/get_string', methods=['GET'])
def get_string():
    global FILEPATH
    global NEW_FILEPATH
    lb=getlb(FILEPATH)
    newlb=getlb(NEW_FILEPATH)
    print(NEW_FILEPATH)
    if lb == newlb:
        str='No'
    else:
        str='Yes'
    data = [
        {
            'score1': lb,
            'score2': newlb,
            'or': str
        }
    ]
    return jsonify(data)


if __name__ == '__main__':
    app.run(debug=True)