from flask import Flask, render_template, request
from flask import g
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd                    # Para la manipulación y análisis de datos
import numpy as np                     # Para crear vectores y matrices n dimensionales
import matplotlib.pyplot as plt        # Para la generación de gráficas a partir de los datos
import seaborn as sns                  # Para la visualización de datos basado en matplotlib

app=Flask(__name__)

class LinkData():
    link = None

LinkDeData = LinkData()

@app.route('/')
def index():
    return render_template('index.html')



@app.route('/menu', methods=["GET", "POST"])
def menu():
    LinkDeData.link = request.form['linkdata']+"?raw=true"
    return render_template('menu.html')

@app.route('/predict', methods=["GET", "POST"])
def predict():
    img_path="app/static/"
    #linkdata = request.form['linkdata']+"?raw=true"
    datosData=pd.read_csv(LinkDeData.link, delimiter=",",index_col=0)
    datosData.hist(figsize=(14,14), xrot=45)
    plt.savefig(img_path+'Histograma.jpg')
    
   

    CorrData = datosData.corr()
    plt.figure(figsize=(20,7))
    sns.heatmap(CorrData, cmap='RdBu_r', annot=True)
    plt.savefig(img_path+'Correlaciones.jpg')

    plt.figure(figsize=(20,7))
    MatrizInf = np.triu(CorrData)
    sns.heatmap(CorrData, cmap='RdBu_r', annot=True, mask=MatrizInf)
    plt.savefig(img_path+'prueba.jpg')

    plt.clf()
    for col in datosData.select_dtypes(include='object'):
        if datosData[col].nunique()<10:sns.countplot(y=col, data=datosData)
        plt.savefig(img_path+'categoricas1.jpg')
    return render_template('result.html')

@app.route('/pca', methods=["GET", "POST"])
def pca():

    img_path="app/static/"
    datosData1=pd.read_csv(LinkDeData.link, delimiter=",",index_col=0)
    CorrData1 = datosData1.corr(method='pearson')
    plt.figure(figsize=(14,7))
    MatrizInf = np.triu(CorrData1)
    sns.heatmap(CorrData1, cmap='RdBu_r', annot=True, mask=MatrizInf)
    plt.savefig(img_path+'CorrPCA.jpg')
    plt.clf()
    
    Estandarizar = StandardScaler()                               # Se instancia el objeto StandardScaler o MinMaxScaler 
    MEstandarizada = Estandarizar.fit_transform(datosData1)
    pd.DataFrame(MEstandarizada, columns=datosData1.columns)
    pca = PCA(n_components=9)     #Se instancia el objeto PCA    #pca=PCA(n_components=None), pca=PCA(.85)
    pca.fit(MEstandarizada)
    Varianza = pca.explained_variance_ratio_

    
    plt.plot(np.cumsum(Varianza))
    plt.xlabel('Número de componentes')
    plt.ylabel('Varianza acumulada')
    plt.grid()
    plt.savefig(img_path+'VarianzaAcumPCA1.jpg')
    
    return render_template('pca.html')

if __name__=='__main__':
    app.run(debug=True, port=5000)