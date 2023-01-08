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
    
    Estandarizar = MinMaxScaler()                               # Se instancia el objeto StandardScaler o MinMaxScaler 
    NuevaMatriz = datosData1.select_dtypes(exclude=['object'])
    MEstandarizada = Estandarizar.fit_transform(NuevaMatriz)
    pd.DataFrame(MEstandarizada, columns=NuevaMatriz.columns)
    #pca = PCA(n_components=9)     #Se instancia el objeto PCA    #pca=PCA(n_components=None), pca=PCA(.85)
    pca = PCA(n_components='mle',svd_solver='full')
    pca.fit(MEstandarizada)
    Varianza = pca.explained_variance_ratio_

    
    plt.plot(np.cumsum(Varianza))
    plt.xlabel('Número de componentes')
    plt.ylabel('Varianza acumulada')
    plt.grid()
    plt.savefig(img_path+'VarianzaAcumPCA1.jpg')
    
    plt.clf()
    sns.pairplot(datosData1, hue = None )
    plt.savefig(img_path+'Visual.jpg')

    return render_template('pca.html')

@app.route('/ADecision', methods=["GET", "POST"])
def ADecision():
    img_path="app/static/"
    datosData2=pd.read_csv(LinkDeData.link, delimiter=",",index_col=0)
    return render_template('ADecision.html')

@app.route('/VerLaData', methods=["GET", "POST"])
def VerLaData():
    VerLaData = pd.read_csv(LinkDeData.link, delimiter=",",index_col=0)
    return VerLaData.to_html()

@app.route('/TiposDeDatos', methods=["GET", "POST"])
def TiposDeDatos():
    TiposDeDatos = pd.read_csv(LinkDeData.link, delimiter=",",index_col=0)
    Tipos = TiposDeDatos.dtypes.to_frame()
    
    return Tipos.to_html()

@app.route('/Nulos', methods=["GET", "POST"])
def Nulos():
    Nulos = pd.read_csv(LinkDeData.link, delimiter=",",index_col=0)
    NulosResult = Nulos.isnull().sum().to_frame() 
    return NulosResult.to_html()

@app.route('/ResumenEstadistico', methods=["GET", "POST"])
def ResumenEstadistico():
    ResumenEstadistico = pd.read_csv(LinkDeData.link, delimiter=",",index_col=0)
    ResumenEstadisticoResult = ResumenEstadistico.describe() 
    return ResumenEstadisticoResult.to_html()

@app.route('/DistribucionCat', methods=["GET", "POST"])
def DistribucionCat():
    DistribucionCat = pd.read_csv(LinkDeData.link, delimiter=",",index_col=0)
    DistribucionCatResult = DistribucionCat.describe(include='object') 
    return DistribucionCatResult.to_html()

@app.route('/MatrizCorr1', methods=["GET", "POST"])
def MatrizCorr1():
    MatrizCorr1 = pd.read_csv(LinkDeData.link, delimiter=",",index_col=0)
    MatrizCorr1Result = MatrizCorr1.corr()
    return MatrizCorr1Result.to_html()

@app.route('/MatrizCorr2', methods=["GET", "POST"])
def MatrizCorr2():
    MatrizCorr2 = pd.read_csv(LinkDeData.link, delimiter=",",index_col=0)
    MatrizCorr2Result = MatrizCorr2.corr(method='pearson')
    return MatrizCorr2Result.to_html()

@app.route('/Estandarizar', methods=["GET", "POST"])
def Estandarizar():
    Data2 = pd.read_csv(LinkDeData.link, delimiter=",",index_col=0)
    Estandarizar = StandardScaler()
    NuevaMatriz = Data2.select_dtypes(exclude=['object'])
    MEstandarizada = Estandarizar.fit_transform(NuevaMatriz)
    Matriz = pd.DataFrame(MEstandarizada, columns=NuevaMatriz.columns) 
    return Matriz.to_html()

@app.route('/Cargas', methods=["GET", "POST"])
def Cargas():
    Data2 = pd.read_csv(LinkDeData.link, delimiter=",",index_col=0)
    Estandarizar = StandardScaler()
    NuevaMatriz = Data2.select_dtypes(exclude=['object'])
    MEstandarizada = Estandarizar.fit_transform(NuevaMatriz)
    pca = PCA(n_components='mle',svd_solver='full')
    pca.fit(MEstandarizada)
    CargasComponentes = pd.DataFrame(abs(pca.components_), columns=NuevaMatriz.columns)
    return CargasComponentes.to_html()

@app.route('/pca/GrafDisper', methods=["GET", "POST"])
def GrafDisper():
    img_path="app/static/"
    Data3 = pd.read_csv(LinkDeData.link, delimiter=",",index_col=0)
    clase1 = request.form['x']
    clase2 = request.form['y']
    clase3 = request.form['hist']
    sns.scatterplot(x=clase1, y =clase2, data=Data3, hue=clase3)
    plt.title('Gráfico de dispersión')
    plt.xlabel(clase1)
    plt.ylabel(clase2)
    plt.savefig(img_path+'GraficoDisper.jpg')
    return render_template('GrafDisper.html')

if __name__=='__main__':
    app.run(debug=True, port=5000)