from flask import Flask, render_template, request
import pandas as pd                    # Para la manipulación y análisis de datos
import numpy as np                     # Para crear vectores y matrices n dimensionales
import matplotlib.pyplot as plt        # Para la generación de gráficas a partir de los datos
import seaborn as sns                  # Para la visualización de datos basado en matplotlib

app=Flask(__name__)

@app.route('/')
def index():
    #return "Hola puto David"
    return render_template('index.html')

@app.route('/predict', methods=["GET", "POST"])
def predict():
    #return "Hola puto David"
    img_path="app/static/"
    linkdata = request.form['linkdata']+"?raw=true"
    datosData=pd.read_csv(linkdata, delimiter=",",index_col=0)
    datosData.hist(figsize=(14,14), xrot=45)
    plt.savefig(img_path+'Histograma.jpg')
    
    CorrData = datosData.corr()
    plt.savefig(img_path+'pito.jpg')

    plt.figure(figsize=(20,7))
    sns.heatmap(CorrData, cmap='RdBu_r', annot=True)
    plt.savefig(img_path+'Correlaciones.jpg')

    plt.figure(figsize=(20,7))
    MatrizInf = np.triu(CorrData)
    sns.heatmap(CorrData, cmap='RdBu_r', annot=True, mask=MatrizInf)
    plt.savefig(img_path+'prueba.jpg')
    return render_template('result.html')

if __name__=='__main__':
    app.run(debug=True, port=5000)