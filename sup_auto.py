from transformers import AutoModel, AutoTokenizer
import torch
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# Carregar o modelo Qwen-2 e o tokenizer
model_name = "Qwen/Qwen2.5-Coder-7B-Instruct"  # Nome do modelo Qwen-2
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# 1. Dados de Perguntas Frequentes (aumentando a quantidade)
data = pd.DataFrame({
    'pergunta': [
        'Como posso alterar minha senha?',
        'Quais são os métodos de pagamento?',
        'Onde encontro o número do pedido?',
        'Como posso rastrear meu pedido?',
        'Como faço para cancelar meu pedido?',
        'Qual é o prazo de entrega?',
        'Como posso falar com o atendimento ao cliente?',
        'Posso alterar meu endereço de entrega após o pedido?',
        'O que faço se recebi um produto errado?',
        'Como posso obter um reembolso?'
    ],
    'resposta': [
        'Para alterar sua senha, vá ao seu perfil e clique em "Alterar Senha".',
        'Aceitamos cartões de crédito, débito e PayPal.',
        'O número do pedido pode ser encontrado no seu e-mail de confirmação.',
        'Você pode rastrear seu pedido no seu painel de controle, seção "Pedidos".',
        'Para cancelar seu pedido, entre em contato com o suporte ao cliente.',
        'O prazo de entrega estimado está disponível na página de confirmação do pedido.',
        'Nosso atendimento ao cliente está disponível via chat e telefone.',
        'Entre em contato com o suporte para verificar a possibilidade de alterar o endereço.',
        'Entre em contato com o suporte para resolver problemas com produtos errados.',
        'Para solicitar um reembolso, acesse sua conta e clique em "Solicitar Reembolso".'
    ]
})

# 2. Função para gerar embeddings com Qwen-2
def generate_embedding(text):
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs)
    embedding = outputs.last_hidden_state.mean(dim=1)  # Média dos embeddings
    return embedding

# Gerar embeddings para perguntas frequentes
data['embedding'] = data['pergunta'].apply(generate_embedding)

# 3. Função para buscar resposta usando similaridade de cosseno
def get_response(question, threshold=0.7):
    question_embedding = generate_embedding(question)
    similarities = [cosine_similarity(question_embedding.detach().numpy(), emb.detach().numpy())[0][0] for emb in data['embedding']]
    
    # Encontra a maior similaridade e seu índice
    max_similarity = max(similarities)
    best_match_index = similarities.index(max_similarity)
    
    if max_similarity >= threshold:
        return data.iloc[best_match_index]['resposta']
    else:
        # Sugere perguntas caso a similaridade seja baixa
        suggestions = "\n".join([f"- {q}" for q in data['pergunta']])
        return f"Não entendi sua pergunta. Você quis dizer uma das seguintes?\n{suggestions}"

# 4. Loop para múltiplas perguntas
while True:
    question = input("Digite sua pergunta (ou 'sair' para encerrar): ")
    if question.lower() == 'sair':
        print("Encerrando o suporte ao cliente. Até mais!")
        break
    response = get_response(question)
    print("Pergunta:", question)
    print("Resposta sugerida:", response)
    print("-" * 50)  # Separador entre perguntas para melhor leitura
