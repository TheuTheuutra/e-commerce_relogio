from sklearn.feature_extraction.text import TfidfVectorizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd

# 1. Dados de Feedback e Avaliações de Clientes
data = pd.DataFrame({
    'feedback': [
        "O relógio é excelente, muito bonito e funciona bem!",
        "A pulseira é desconfortável e a bateria dura pouco.",
        "Gostei bastante, mas o preço é um pouco alto.",
        "Produto de qualidade ruim. Não recomendo.",
        "Excelente custo-benefício, recomendo a todos.",
        "O visor risca com facilidade, esperava mais durabilidade."
    ]
})

# 2. Limpeza dos Dados (removendo palavras irrelevantes)
stop_words = ['o', 'a', 'e', 'é', 'um', 'uma', 'de', 'do', 'da']  # palavras irrelevantes comuns
data['feedback_limpo'] = data['feedback'].str.lower().replace('|'.join(stop_words), '', regex=True)

# 3. Vetorização TF-IDF para identificar palavras mais importantes
tfidf = TfidfVectorizer(stop_words=stop_words)
tfidf_matrix = tfidf.fit_transform(data['feedback_limpo'])
tfidf_feature_names = tfidf.get_feature_names_out()

# 4. Análise de Sentimentos com VADER
analyzer = SentimentIntensityAnalyzer()

def analyze_sentiment(text):
    scores = analyzer.polarity_scores(text)
    if scores['compound'] >= 0.05:
        return 'Positivo'
    elif scores['compound'] <= -0.05:
        return 'Negativo'
    else:
        return 'Neutro'

# Análise de sentimento para cada feedback
data['sentimento'] = data['feedback'].apply(analyze_sentiment)

# 5. Loop para Análise e Sugestões
while True:
    user_feedback = input("Digite um feedback de cliente (ou 'sair' para encerrar): ")
    if user_feedback.lower() == 'sair':
        print("Encerrando análise de feedback. Até mais!")
        break
    
    # Análise de sentimento do feedback
    sentiment = analyze_sentiment(user_feedback)
    
    # Vetorização do feedback do usuário com TF-IDF para identificar palavras-chave
    user_feedback_cleaned = ' '.join([word for word in user_feedback.lower().split() if word not in stop_words])
    tfidf_user = tfidf.transform([user_feedback_cleaned])
    keywords = [tfidf_feature_names[i] for i in tfidf_user.indices]
    
    print("Sentimento detectado:", sentiment)
    print("Palavras-chave identificadas no feedback:", ', '.join(keywords))
    print("Sugestões de melhorias com base no feedback:")
    
    if sentiment == 'Negativo':
        print("- Avalie a durabilidade e o conforto do produto.")
        print("- Considere aprimorar a bateria e o acabamento.")
    elif sentiment == 'Positivo':
        print("- Continue mantendo a qualidade!")
        print("- Avalie se outros aspectos elogiados podem ser replicados em outros modelos.")
    else:
        print("- Considere o feedback neutro para ajustes menores.")
    
    print("-" * 50)  # Separador entre as análises