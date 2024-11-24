from transformers import AutoModel, AutoTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np

# Carregar o modelo Qwen-2 e o tokenizer
model_name = "Qwen/Qwen2.5-Coder-7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Stop words em português
stop_words_pt = ["o", "a", "e", "de", "do", "da", "para", "em", "um", "uma", "por", "com", "não", "que", "os", "as"]

# 1. Coleta e Pré-processamento dos Dados
data = pd.DataFrame({
    'pedido': [
        'Solicitação para envio expresso de relógio.',
        'Pedido de devolução de produto incorreto.',
        'Pedido para cancelar a compra devido a erro no pagamento.',
        'Requisição de reembolso por item com defeito.',
        'Pedido incompleto sem endereço de entrega.'
    ],
    'status': [
        'processar', 'rejeitar', 'rejeitar', 'rejeitar', 'rejeitar'
    ],
    'motivo_rejeicao': [
        None,
        'Produto incorreto mencionado.',
        'Erro no pagamento.',
        'Defeito no item.',
        'Informação insuficiente.'
    ]
})

# 2. Representação do Texto com TF-IDF usando stop words personalizadas
tfidf_vectorizer = TfidfVectorizer(stop_words=stop_words_pt)
tfidf_matrix = tfidf_vectorizer.fit_transform(data['pedido'])

# Função para gerar embeddings com Qwen-2
def generate_embedding(text):
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs)
    embedding = outputs.last_hidden_state.mean(dim=1)  # Média dos embeddings
    return embedding

# Gerar embeddings para cada pedido
data['embedding'] = data['pedido'].apply(generate_embedding)

# 3. Função para Verificação Automática de Pedidos
def verificar_pedido(pedido_texto):
    pedido_embedding = generate_embedding(pedido_texto)
    similarities = [
        cosine_similarity(pedido_embedding.detach().numpy(), emb.detach().numpy())[0][0]
        for emb in data['embedding']
    ]
    
    # Identificar o índice do pedido mais similar
    best_match_index = similarities.index(max(similarities))
    
    # Retornar o status e, se houver, o motivo da rejeição
    status = data.iloc[best_match_index]['status']
    motivo_rejeicao = data.iloc[best_match_index]['motivo_rejeicao']
    
    # Exibir resultados
    if status == "rejeitar" and motivo_rejeicao:
        return f"Pedido rejeitado. Motivo: {motivo_rejeicao}"
    elif status == "processar":
        return "Pedido aprovado para processamento."
    else:
        return "Status do pedido não identificado."

# 4. Exemplo de uso
while True:
    pedido_usuario = input("Digite o detalhe do pedido (ou 'sair' para encerrar): ")
    if pedido_usuario.lower() == 'sair':
        break
    
    resultado = verificar_pedido(pedido_usuario)
    print("Resultado:", resultado)
