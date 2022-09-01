from flask import Flask, request, render_template
from Iris import iris


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'temp_upload'


def iris_predict(sepal_length: float, sepal_width: float, petal_length: float, petal_width: float) -> str:
    ir = iris.Iris()
    predition = ir.predict(sepal_length,sepal_width,petal_length,petal_width)
    return predition

@app.route("/iris",methods=["GET","POST"])
def iris_controller():
    if request.method == 'POST':
        sepal_length = float(request.form.get('sepal_length'))
        sepal_width = float(request.form.get('sepal_width'))
        petal_length = float(request.form.get('petal_length'))
        petal_width = float(request.form.get('petal_width'))
        prediction = iris_predict(sepal_length,sepal_width,petal_length,petal_width)
        return prediction
    return render_template("iris.html")

@app.route("/mnist",methods=["GET","POST"])
def mnist_controller():
    if request.method == 'POST':
        file = request.files.get('image')
        filename = file.filename
        file_path = "temp_upload/"+filename
        file.save("images/"+filename)
        return "File recieved"

        
    return  render_template("mnist.html")

app.run(host="127.0.0.1",port=5000,debug=True)
