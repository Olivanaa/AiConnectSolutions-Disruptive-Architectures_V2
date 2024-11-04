import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from keras.models import Sequential
from keras.layers import Dense

from sklearn.cluster import KMeans
import sqlalchemy

import warnings
warnings.filterwarnings('ignore')

# Função para carregar dados do banco de dados
def carregar_dados_banco(dialeto, usuario, senha, endereco, porta, nome_do_banco, tabela):
    engine = sqlalchemy.create_engine(f'{dialeto}://{usuario}:{senha}@{endereco}:{porta}/{nome_do_banco}')
    return pd.read_sql(f'SELECT * FROM {tabela}', engine)

# Função para executar KMeans e retornar os clusters
def clusterizar_dados(df, n_clusters=4):
    df_cluster = df.drop(columns=['ID'])
    modelo_kmeans = KMeans(n_clusters=n_clusters)
    df['Classificação_cliente'] = modelo_kmeans.fit_predict(df_cluster)  
    return df

# Função para treinar o modelo de classificação
def treinar_modelo(X_train, y_train):
    model = Sequential()
    model.add(Dense(10, input_shape=(X_train.shape[1],), activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(y_train.shape[1], activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=100)
    return model

# Função para atualizar o banco de dados com a classificação
def atualizar_classificacao(engine, df, nome_da_tabela):
    with engine.connect() as connection:
        for index, row in df.iterrows():
            query = f"""
            UPDATE {nome_da_tabela}
            SET Classificação_cliente = :classificacao_cliente, Classificação_lead = :classificacao_lead
            WHERE ID = :id
            """
            connection.execute(query, {
                'classificacao_cliente': row['Classificação_cliente'], 
                'classificacao_lead': row['Classificação_lead'], 
                'id': row['ID']
            })

# Função principal
def main():
    # Carregar dados do banco de dados
    dialeto = 'oracle'
    usuario = 'usuario'
    senha = 'senha'
    endereco = 'endereco'
    porta = 'porta'
    nome_do_banco = 'nome_do_banco'

    tabela = input("Digite o nome da tabela que deseja carregar: ")

    df = carregar_dados_banco(dialeto, usuario, senha, endereco, porta, nome_do_banco, tabela)

    print("Dados carregados com sucesso!")
    print(df.head())

    # Realizar a clusterização dos dados
    df = clusterizar_dados(df)

    # Preparar dados para treinamento do modelo
    X = df.drop(columns=['Classificação_cliente', 'ID'])
    y = df['Classificação_cliente'].values.reshape(-1, 1)
    encoder = OneHotEncoder(sparse=False)
    y_encoded = encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.20, random_state=27)

    # Treinar o modelo
    modelo = treinar_modelo(X_train, y_train)

    # Fazer previsões
    previsoes = modelo.predict(X)
    df['Previsão'] = np.argmax(previsoes, axis=1)

    # Avaliar o modelo
    loss, accuracy = modelo.evaluate(X_test, y_test)
    print(f'Acurácia do modelo: {accuracy}')

    # Atualizar classificação no banco de dados
    engine = sqlalchemy.create_engine(f'{dialeto}://{usuario}:{senha}@{endereco}:{porta}/{nome_do_banco}')
    atualizar_classificacao(engine, df, tabela)

# Executar a função principal
if __name__ == "__main__":
    main()