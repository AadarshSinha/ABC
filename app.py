from flask import Flask,render_template,request,jsonify
import pickle
from PIL import Image
import numpy as np
model = pickle.load(open("model3.pkl", "rb"))
app=Flask(__name__)

@app.route('/',methods=['GET'])
def Hi():
    return render_template('index.html')

@app.route('/',methods=['POST'])
def Render():
    print("got the request")
    print("request = ",request.files.lists)
    print("request = ",request.files["imageFile"])
    image=request.files['imageFile']
    path="./images/" + image.filename
    image.save(path)
    img_pil = Image.open(path).convert('L')
    img_28x28 = np.array(img_pil.resize((28, 28), Image.Resampling.LANCZOS))
    img_array = (img_28x28.flatten())
    img_array  = img_array.reshape(-1,1).T
    ans=model.predict(np.array(img_array))
    print("input = ",np.array(img_array).shape)
    print("result = ",int(ans[0]))
    result={
        "result":int(ans[0]),
    }
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
