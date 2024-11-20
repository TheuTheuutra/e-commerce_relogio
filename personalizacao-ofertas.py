import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import nltk
from nltk.corpus import stopwords
import re
import torch
from transformers import AutoTokenizer, AutoModel

# Configuração para Qwen-2
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-2")
model = AutoModel.from_pretrained("Qwen/Qwen-2")

# Baixar stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('portuguese'))

# Função de Limpeza de Texto
def limpar_texto(texto):
    texto = re.sub(r'\W', ' ', texto)
    texto = texto.lower()
    texto = ' '.join([palavra for palavra in texto.split() if palavra not in stop_words])
    return texto

# Carregar e Processar Dados
df = pd.read_csv('dados.csv')
df['texto_limpo'] = df['descricao'].apply(limpar_texto)

# Função para Obter Embeddings com Qwen-2
def obter_embeddings(texto):
    inputs = tokenizer(texto, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state.mean(dim=1).squeeze()
    return embeddings.cpu().numpy()

# Obter Embeddings de Todos os Textos
df['embeddings'] = df['texto_limpo'].apply(obter_embeddings)
X = np.vstack(df['embeddings'].values)

# Definir Variável Alvo
y = df['preferencia_cliente']  # Ajuste com a coluna que representa a preferência

# Dividir Dados para Treinamento e Teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Treinar o Modelo com Random Forest
modelo = RandomForestClassifier()
modelo.fit(X_train, y_train)

# Avaliar o Modelo
y_pred = modelo.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Acurácia do Modelo: {accuracy * 100:.2f}%')
