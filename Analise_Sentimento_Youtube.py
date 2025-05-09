from googleapiclient.discovery import build
import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
import unicodedata
from wordcloud import WordCloud

# Chave e URL do vídeo
key = 'AIzaSyAVUKeywyHi-26sI7WYknNe-ambU7p904A'
url = 'https://www.youtube.com/watch?v=mJCRLfac3G4'  # Substitua pelo URL do vídeo

# Download de recursos do NLTK
nltk.download('stopwords')
nltk.download('punkt')

# Stopwords
stop_words_pt = set(stopwords.words('portuguese'))

# Função para remover acentos
def remover_acentos(texto):
    return ''.join(c for c in unicodedata.normalize('NFD', texto)
                   if unicodedata.category(c) != 'Mn')

# Função para pré-processar texto
def preprocessar_texto(texto):
    texto = re.sub(r'http\S+|@\S+|#\S+', '', texto)
    texto = remover_acentos(texto.lower())
    tokens = re.findall(r'\b\w+\b', texto)
    tokens = [w for w in tokens if w.isalpha() and w not in stop_words_pt]
    return tokens

# Função para extrair comentários com paginação
def extrair_comentarios_youtube(video_url, api_key, max_total=300):
    match = re.search(r"v=([a-zA-Z0-9_-]+)", video_url)
    if not match:
        raise ValueError("URL do vídeo inválida")
    video_id = match.group(1)
    youtube = build('youtube', 'v3', developerKey=api_key)
    
    comentarios = []
    next_page_token = None
    while len(comentarios) < max_total:
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=min(100, max_total - len(comentarios)),
            textFormat="plainText",
            pageToken=next_page_token
        )
        response = request.execute()
        for item in response['items']:
            comentario = item['snippet']['topLevelComment']['snippet']['textDisplay']
            comentarios.append(comentario)
        next_page_token = response.get('nextPageToken')
        if not next_page_token:
            break
    return comentarios

# Léxico de sentimentos
lexico_sentimentos = {
    # Positivos
    "revolucionario": 1, "revolucionaria": 1, "revolucionarios": 1,
    "inovador": 1, "inovadora": 1, "inovadores": 1, "inovar": 1,
    "impressionante": 1, "impressionantes": 1, "desespero": 1,
    "fascinante": 1, "fascinantes": 1, "muito bom": 1,
    "transformador": 1, "transformadora": 1, "transformadores": 1, "transformar": 1,
    "inteligente": 1, "inteligentes": 1, "amigo": 1,
    "promissor": 1, "promissora": 1,
    "extraordinario": 1, "boa": 1, "extraordinaria": 1, "parabens": 1,
    "eficiente": 1, "eficientes": 1,"ótimo": 1, "otima": 1,
    "inspirador": 1, "inspiradora": 1, "inspiradoramente": 1,
    "poderoso": 1, "poderosa": 1, "adorei": 1,
    "brilhante": 1, "brilhantes": 1, "apaixonado": 1,
    "util": 1, "uteis": 1, "gostei": 1, "otimista": 1,
    "avancado": 1, "avancada": 1, "avancados": 1,
    "surpreendente": 1, "curioso": 1, "passe logo": 1,
    "legal": 1, "legais": 1, "bom": 1,
    "top": 1, "massa": 1, "show": 1, "incrivel": 1,
    "perfeito": 1, "perfeita": 1, "excelente": 1, "excelentes": 1,
    "foda": 1, "f#da": 1, "fodastico": 1, "monstro": 1, "gênio": 1,
    "esperanca": 1, "esperanca": 1, "beleza": 1,
    "essencial": 1, "essenciais": 1,
    "😊": 1, "😍": 1, "👏": 1, "🔥": 1, "💪": 1, "😄": 1, "❤️❤️❤️❤️❤️": 1,

    # Neutros
    "neutro": 0, "interessante": 0, "interessantes": 0,
    "tendencia": 0, "tendencias": 0,
    "comum": 0, "comuns": 0,
    "razoavel": 0, "razoaveis": 0,
    "funcional": 0, "funcionais": 0,
    "analitico": 0, "analitica": 0,
    "meh": 0, "ok": 0, "tanto faz": 0, "depende": 0, "mais ou menos": 0,
    "😐": 0, "🤔": 0,

    # Negativos
    "assustador": -1, "assustadora": -1, "transbordar": -1, "Falta segurança": -1, "coitado": -1,
    "perigoso": -1, "perigosa": -1, "perigosos": -1, "pontos de alagamento": -1,
    "problematico": -1, "problematica": -1, "sem infraestrutura": -1, "pena": -1,
    "nao confiavel": -1, "inconfiavel": -1, "transporte publico": -1, "engarrafamento": -1,
    "ameaçador": -1, "ameacadora": -1, "dificil": -1, "situacao dificil": -1,
    "invasivo": -1, "invasiva": -1, "alagamentos": -1, "alagamento": -1, "ricos": -1,
    "morte": -1, "mortes": -1, "tristeza": -1, "meu deus": -1,
    "exagerado": -1, "exagerada": -1, "Misericórdia": -1, "políticos": -1, "político": -1, 
    "preocupante": -1, "mare alta": -1, "maré alta": -1,"situação caótica": -1,
    "manipulador": -1, "manipuladora": -1, "agua": -1, "muita agua": -1, 
    "ineficiente": -1, "chuva": -1, "planejamento": -1, "pessima gestao": -1,
    "limitado": -1, "limitada": -1,"nao para": -1, "drastica": -1, " nao vai mudar": -1,
    "erroneo": -1, "errado": -1, "equivocado": -1, "carnaval": -1,
    "impessoal": -1, "morreria": -1, "falta de": -1, "descraça": -1,
    "prejudicial": -1, "dominados": -1, "chuva": -1, "problema": -1,
    "inferior": -1, "doutrinados": -1, "transtornos": -1,
    "desmatamento": -1, "devastacao": -1, "devastador": -1,
    "urgente": -1, "urgencia": -1, "nunca vai mudar": -1,
    "crise": -1, "crises": -1, "carro": -1, "carros": -1, 
    "destruicao": -1, "destruidor": -1, "prejuizo": -1,
    "negativo": -1, "negativa": -1, "negativos": -1,
    "problema": -1, "problemas": -1, "não": -1, "nao gostou": -1,
    "ruim": -1, "ruins": -1, "ruins": -1, "ruins": -1, "estresse": -1,
    "ruim": -1, "pessimo": -1, "pessima": -1, "lixo": -1, "raiva": -1,
    "horrivel": -1, "detestei": -1, "odiei": -1, "desemprego": -1,
    "🤮": -1, "😡": -1, "👎": -1, "💩": -1,

    # Gírias e internetês (positivos)
    "show de bola": 1, "topzera": 1, "top demais": 1,"haha": 1,
    "maneiro": 1, "massa demais": 1, "sensacional": 1,
    "bom demais": 1, "daora": 1, "da hora": 1, "daora demais": 1,
    "kkk": 1, "kkkk": 1, "hahaha": 1, "rsrs": 1, "like": 1, "curtir": 1,
    "ameiii": 1,

    # Gírias e internetês (negativos)
    "aff": -1, "argh": -1, "zzz": -1, "chato": -1, "chata": -1,
    "bosta": -1, "merda": -1, "burro": -1, "burra": -1
}

# Função para análise de sentimento
def analisar_sentimento(texto_processado, lexico):
    pontuacao = 0
    num_palavras = 0
    for palavra in texto_processado:
        if palavra in lexico:
            pontuacao += lexico[palavra]
            num_palavras += 1
    if num_palavras > 0:
        sentimento = pontuacao / num_palavras
        if sentimento > 0.2:
            return "positivo"
        elif sentimento < -0.2:
            return "negativo"
        else:
            return "neutro"
    else:
        return "neutro"

# Executar análise completa
comentarios = extrair_comentarios_youtube(url, key)
df = pd.DataFrame(comentarios, columns=["comentario"])

resultados = []
for comentario in df["comentario"]:
    tokens = preprocessar_texto(comentario)
    sentimento = analisar_sentimento(tokens, lexico_sentimentos)
    resultados.append(sentimento)
    print(f"Comentário: {comentario}\n-> Sentimento: {sentimento}\n")

df["sentimento"] = resultados

# Contagem dos sentimentos
contagem_sentimentos = df["sentimento"].value_counts().to_dict()

print("Contagem geral de sentimentos:")
print(contagem_sentimentos)

# Juntar todos os comentários processados
todos_tokens = []

for comentario in df["comentario"]:
    tokens = preprocessar_texto(comentario)
    todos_tokens.extend(tokens)

# Unir tudo em um texto
texto_unido = " ".join(todos_tokens)

# Gerar a nuvem de palavras
wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='viridis').generate(texto_unido)

# Mostrar a nuvem de palavras
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

wordcloud = WordCloud(
    width=800,
    height=400,
    background_color='black',
    colormap='Set2', # cores diferentes!
).generate(texto_unido)

# Gráfico de barras
plt.figure(figsize=(6, 4))
plt.bar(contagem_sentimentos.keys(), contagem_sentimentos.values(), color=['gray', 'red', 'green'])
plt.title('Distribuição de Sentimentos dos Comentários')
plt.xlabel('Sentimento')
plt.ylabel('Quantidade')
plt.tight_layout()
plt.show()
