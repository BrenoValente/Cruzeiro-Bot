import tweepy
import datetime
from datetime import timezone, timedelta
import pandas as pd
import schedule
import time
import pytz
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from wordcloud import WordCloud, ImageColorGenerator
import nltk
import boto
import boto3
from boto.s3.key import Key

consumer_key = CONSUMER_KEY
consumer_secret = CONSUMER_SECRET
access_token = ACCESS_KEY
access_token_secret= ACCESS_SECRET

key = AMAZON_KEY
secret = AMAZON_SECRET

def data_dia():
  #função para
  return datetime.datetime.now().replace(tzinfo=pytz.UTC)-datetime.timedelta(1)


#função para autenticar o usuário. toda vez tem que fazer. retorna um objeto api
def autenticar():
  auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
  auth.set_access_token(access_token, access_token_secret)

  return(tweepy.API(auth))



def ler_dataframe(nome):

  s3 = boto3.client('s3', aws_access_key_id=key, aws_secret_access_key=secret) 
  obj = s3.get_object(Bucket= 'cruzeirobot', Key=nome)
  initial_df = pd.read_csv(obj['Body']) # 'Body' is a key word

  return initial_df


def remover_dataframe(nome:str = 'myfile'):

  s3 = boto3.resource('s3', aws_access_key_id=key, aws_secret_access_key=secret)
  s3.Object('cruzeirobot', nome).delete()

def upload_dataframe(df, nome = 'df_geral_diario'):
  df.to_csv(f'{nome}.csv')
  c = boto.connect_s3(key, secret)
  b = c.get_bucket('cruzeirobot') # substitute your bucket name here
  k = Key(b)
  k.key = nome
  time.sleep(2)
  k.set_contents_from_filename(f'{nome}.csv')


#função para pegar o número de seguidores do perfil oficial. precisa do objeto api
def numero_seguidores(api):
  seguidores = api.get_user(screen_name='Cruzeiro').followers_count
  return seguidores


#função para pegar os tweets do perfil oficial. só pega a partir do último pego na última atualização
def extrai_tweets(api, since_id = None):
  tweets = api.user_timeline(screen_name='Cruzeiro', 
                           include_rts = True, 
                           count = 200,
                           # Necessary to keep full_text 
                           # otherwise only the first 140 words are extracted
                           )
  return tweets

def filtra_tweets(tweets_lista, ponto_corte):
  nova_lista = []
  for tweet in tweets_lista:
    if tweet.created_at >= ponto_corte:
      nova_lista.append(tweet)
  
  return nova_lista

def extrai_num_favs(tweets):
  lista_contagem = [tweet.favorite_count for tweet in tweets]
  num_fav = sum(lista_contagem)

  return num_fav



def extrai_num_rts(tweets):
  lista_contagem = [tweet.retweet_count for tweet in tweets]
  num_rts = sum(lista_contagem)

  return num_rts



def info_tweet(tweets, lista_corrente = []):

  for tweet in tweets:
    l = []
    l.append(tweet.id)
    l.append(tweet.created_at)
    l.append(tweet.text)
    l.append(tweet.favorite_count)
    l.append(tweet.retweet_count)
    l.append(numero_seguidores)
    l.append(tweet.entities['hashtags'])
    l.append(tweet.entities['symbols'])
    l.append(tweet.entities['urls'])
    l.append(tweet.entities['user_mentions'])

    lista_corrente.append(l)

  return lista_corrente

def formata(num):
  if num > 0:
    palavra = str(num)
    palavra = list(palavra)
    for i,j in enumerate(palavra[::-1]):
      if (i+1)%3 == 0:
        if len(palavra)-i-1 != 0:
          palavra[len(palavra)-i-1] = f'.{palavra[len(palavra)-i-1]}'
    
  
  else:
    palavra = str(num*(-1))
    palavra = list(palavra)
    for i,j in enumerate(palavra[::-1]):
      if (i+1)%3 == 0:
        if len(palavra)-i-1 != 0:
          palavra[len(palavra)-i-1] = f'.{palavra[len(palavra)-i-1]}'
    palavra[0] = f'-{palavra[0]}'
  
  return ''.join(palavra)

def sinal(var):
  
  if var > 0:
    sinal = '+'
  elif var <= 0:
    sinal = ''
  else:
    sinal = ''
  return(sinal)

def converte_data(x):
  data = str(x)[:10]
  data = pd.to_datetime(data)

  return data

def cria_grafico(df, indicador = 0, nome = 'interacoes_semana.png'):
  """
  indicador = 0 -> Semana
  indicador = 1 -> Mês
  """
  if indicador == 0:
    txt_complemento = 'NA SEMANA'
  else:
    txt_complemento = 'NO MÊS'

  lista = list(df.index)
  lim_inf = int(lista[0].strftime('%d'))
  lim_sup = int(lista[-1].strftime('%d'))

  x = [i for i in range(lim_inf, lim_sup+1)]
  y = [int(i) for i in list(df.interacoes)]

  maior_y = 0
  for i, j in enumerate(y):
    if i == 1:
      maior_y = i
    else:
      if j > y[maior_y]:
        maior_y = i 

  fig, ax = plt.subplots(figsize = (9,4.5))
  ax.plot(x, y, linewidth=3.5, color = '#1e3d8f')

  ax.spines['right'].set_visible(False)
  ax.spines['top'].set_visible(False)

  ax.set_xlabel('DIA', fontsize = 12)
  ax.set_ylabel('Nº de Interações')

  ax.set_ylim([0, y[maior_y]*1.15])

  ax.set_xticks(x)

  im = Image.open('escudo.png')
  fig.figimage(im, 325, 100, zorder=3, alpha=.1)

  plt.text(x = x[0]-.3, y = y[maior_y]*1.2257,s = f'INTERAÇÕES {txt_complemento}', fontsize = 18, weight = 'bold', color = '#000033')
  ax.scatter(x = x[maior_y], y = y[maior_y], s = 150)
  plt.text(x = x[maior_y]+.25, y = y[maior_y], s = f"{formata(int(y[maior_y]))}", fontsize = 15)
  
  plt.savefig(nome, bbox_inches='tight')




def cria_tweet_diferencas(seguidores, diferenca_seguidores, tweets, diferenca_tweets, favoritos, diferenca_favoritos, rts, diferenca_rts):
  texto = f'👥Número de seguidores: {formata(seguidores)} ({sinal(seguidores)}{formata(diferenca_seguidores)})\n\n🦊 Número de tweets: {formata(tweets)} ({sinal(diferenca_tweets)}{formata(diferenca_tweets)})\n💙 Número de favoritos: {formata(favoritos)} ({sinal(diferenca_favoritos)}{formata(diferenca_favoritos)})\n🔃 Número de retweets: {formata(rts)} ({sinal(diferenca_rts)}{formata(diferenca_rts)})\n\n#FechadoComOCruzeiro\n@Cruzeiro'

  return texto


def cria_tweet_dia(data,tweets, favoritos, rts, diferenca_tweets, diferenca_favoritos, diferenca_rts):
  dia = data.strftime('%d/%m')
  print(dia)
  texto = f'Tweets no dia {dia}\n\n🦊 Número de tweets: {formata(tweets)} ({sinal(diferenca_tweets)}{formata(diferenca_tweets)})\n💙 Número de favoritos: {formata(favoritos)} ({sinal(diferenca_favoritos)}{formata(diferenca_favoritos)})\n🔃 Número de retweets: {formata(rts)} ({sinal(diferenca_rts)}{formata(diferenca_rts)})\n\n#FechadoComOCruzeiro\n@Cruzeiro'

  return texto


def cria_tweet_semana(dias_atras,tweets, favoritos, rts, diferenca_tweets, diferenca_favoritos, diferenca_rts):
  texto = f'Tweets nos últimos {dias_atras} dias\n\n🦊 Número de tweets: {formata(tweets)} ({sinal(diferenca_tweets)}{formata(diferenca_tweets)})\n💙 Número de favoritos: {formata(favoritos)} ({sinal(diferenca_favoritos)}{formata(diferenca_favoritos)})\n🔃 Número de retweets: {formata(rts)} ({sinal(diferenca_rts)}{formata(diferenca_rts)})\n\n#FechadoComOCruzeiro\n@Cruzeiro'

  return texto


def atualizacao_hora(num_horas = 3):
  global api, numero_seguidores, diferenca_seguidores, tweets, tweets_filtrados, numero_tweets, diferenca_tweets, numero_likes, diferenca_likes
  global numero_rts, diferenca_rts, lista_info_tweets, lista_info_df, texto_tweet, seguidores_antigo, tweets_antigo, likes_antigo, rts_antigo


  data_corte_hora = datetime.datetime.now().replace(tzinfo=pytz.UTC)-datetime.timedelta(hours = num_horas)

  api = autenticar()


  tweets = extrai_tweets(api)
  tweets_filtrados = filtra_tweets(tweets, data_corte_hora)
  if len(tweets_filtrados) != 0:

    numero_tweets = len(tweets_filtrados)
    diferenca_tweets = int(numero_tweets - tweets_antigo)

    num_seguidores = numero_seguidores(api)
    diferenca_seguidores = int(num_seguidores - seguidores_antigo)



    numero_likes = extrai_num_favs(tweets_filtrados)
    diferenca_likes = int(numero_likes - likes_antigo)

    numero_rts = extrai_num_rts(tweets_filtrados)
    diferenca_rts = int(numero_rts - rts_antigo)


    texto_tweet = cria_tweet_diferencas(num_seguidores, diferenca_seguidores, numero_tweets,diferenca_tweets, numero_likes, diferenca_likes, numero_rts, diferenca_rts)


    seguidores_antigo = num_seguidores
    tweets_antigo = len(tweets_filtrados)
    likes_antigo = numero_likes
    rts_antigo = numero_rts
    
    api.update_status(texto_tweet)

    
  else:
    texto_tweet = f'O perfil oficial do Cruzeiro não publicou tweets nas últimas {num_horas} horas.\n\n#FechadoComOCruzeiro\n@Cruzeiro'
      
  

def atualizacao_dia(dia):
  
  global api, numero_seguidores, diferenca_seguidores, tweets, tweets_filtrados, numero_tweets, diferenca_tweets_dia, numero_likes, diferenca_likes_dia
  global numero_rts, diferenca_rts_dia, lista_info_tweets, lista_info_df, texto_tweet, seguidores_antigo, tweets_antigo_dia, likes_antigo_dia, rts_antigo_dia
  

  #atualizacao_dia, dia = datetime.datetime.now().replace(tzinfo=pytz.UTC)-datetime.timedelta(hours = 4)
  
  data_corte_hora = datetime.datetime.now().replace(tzinfo=pytz.UTC)-datetime.timedelta(hours = 24)

  api = autenticar()



  tweets = extrai_tweets(api)
  tweets_filtrados = filtra_tweets(tweets, data_corte_hora)
  numero_tweets = len(tweets_filtrados)
  diferenca_tweets = int(numero_tweets - tweets_antigo_dia)

  numero_likes = extrai_num_favs(tweets_filtrados)
  diferenca_likes = int(numero_likes - likes_antigo_dia)

  numero_rts = extrai_num_rts(tweets_filtrados)
  diferenca_rts = int(numero_rts - rts_antigo_dia)


  lista_info_tweets = info_tweet(tweets_filtrados)
  lista_info_df = pd.DataFrame(lista_info_tweets, columns = ['ID', 'data_criacao', 'texto', 'num_favs', 'num_rts', 'Seguidores', 'hashtags', 'simbolos', 'urls', 'mencoes'])

  #ler o antigo
  data_ultimo = ler_dataframe("geral_tweets")
  data_ultimo = data_ultimo[['ID', 'data_criacao', 'texto', 'num_favs', 'num_rts', 'Seguidores', 'hashtags', 'simbolos', 'urls', 'mencoes']]
  data_ultimo = data_ultimo.drop_duplicates('ID', inplace = True, keep = 'last')
  #atualizar o antigo
  data_ultimo = pd.concat([data_ultimo, lista_info_df])
  #remover o antigo
  remover_dataframe("geral_tweets")
  #subir o novo
  upload_dataframe(data_ultimo, nome = 'geral_tweets')


  data = dia
  texto_tweet_sem_dif = cria_tweet_dia(data,numero_tweets, numero_likes, numero_rts, diferenca_tweets, diferenca_likes, diferenca_rts)

  tweets_antigo_dia = numero_tweets
  likes_antigo_dia = numero_likes
  rts_antigo_dia = numero_rts

  api.update_status(texto_tweet_sem_dif)


def atualizacao_semana():
  global api, numero_seguidores, diferenca_seguidores, tweets, tweets_filtrados, numero_tweets, diferenca_tweets_semana, numero_likes, diferenca_likes_semana
  global numero_rts, diferenca_rts_semana, lista_info_tweets, lista_info_df, texto_tweet, seguidores_antigo_semana, tweets_antigo_semana, likes_antigo_semana, rts_antigo_semana
  


  data_corte_dia = datetime.datetime.now().replace(tzinfo=pytz.UTC)-datetime.timedelta(days = 7)
  limite_superior = datetime.datetime.now().replace(tzinfo=pytz.UTC)-datetime.timedelta(days = 1)
  

  api = autenticar()


  tweets = extrai_tweets(api)
  tweets_filtrados = filtra_tweets(tweets, data_corte_dia)

  nova_lista = []
  for tweet in tweets_filtrados:
    if tweet.created_at <= limite_superior:
      nova_lista.append(tweet)
  
  tweets_filtrados = nova_lista


  numero_tweets = len(tweets_filtrados)
  diferenca_tweets = int(numero_tweets - tweets_antigo_semana)

  numero_likes = extrai_num_favs(tweets_filtrados)
  diferenca_likes = int(numero_likes - likes_antigo_semana)

  numero_rts = extrai_num_rts(tweets_filtrados)
  diferenca_rts = int(numero_rts - rts_antigo_semana)

  lista_info_tweets = info_tweet(tweets_filtrados)
  lista_info_df = pd.DataFrame(lista_info_tweets, columns = ['ID', 'data_criacao', 'texto', 'num_favs', 'num_rts', 'Seguidores', 'hashtags', 'simbolos', 'urls', 'mencoes'])

  #ler o antigo
  data_ultimo = ler_dataframe('geral_tweets_semanal')
  data_ultimo = data_ultimo[['ID', 'data_criacao', 'texto', 'num_favs', 'num_rts', 'Seguidores', 'hashtags', 'simbolos', 'urls', 'mencoes']]
  data_ultimo = data_ultimo.drop_duplicates('ID', inplace = True, keep = 'last')

  #atualizar o antigo
  data_ultimo = pd.concat([data_ultimo, lista_info_df])

  #remover o antigo
  remover_dataframe("geral_tweets_semanal")

  #subir o novo
  upload_dataframe(data_ultimo, nome = 'geral_tweets_semanal')

  texto_tweet = cria_tweet_semana(7,numero_tweets, numero_likes, numero_rts, diferenca_tweets, diferenca_likes, diferenca_rts)


  tweets_antigo_semana = len(tweets_filtrados)
  likes_antigo_semana = numero_likes
  rts_antigo_semana = numero_rts


  api.update_status(texto_tweet)


def atualizacao_semana_limpo():
  global api, numero_seguidores, diferenca_seguidores, tweets, tweets_filtrados, numero_tweets, diferenca_tweets, numero_likes, diferenca_likes
  global numero_rts, diferenca_rts, lista_info_tweets, lista_info_df, texto_tweet, seguidores_antigo, tweets_antigo, likes_antigo
  


  data_corte_dia = datetime.datetime.now().replace(tzinfo=pytz.UTC)-datetime.timedelta(days = 7)
  limite_superior = datetime.datetime.now().replace(tzinfo=pytz.UTC)-datetime.timedelta(days = 1)
  

  api = autenticar()

  tweets_filtrados = filtra_tweets(tweets, data_corte_dia)

  nova_lista = []
  for tweet in tweets_filtrados:
    if tweet.created_at <= limite_superior:
      nova_lista.append(tweet)
  
  tweets_filtrados = nova_lista

  lista_info_tweets = info_tweet(tweets_filtrados)
  lista_info_df = pd.DataFrame(lista_info_tweets, columns = ['ID', 'data_criacao', 'texto', 'num_favs', 'num_rts', 'Seguidores', 'hashtags', 'simbolos', 'urls', 'mencoes'])

  #ler o antigo
  data_ultimo = ler_dataframe('geral_tweets_semanal')
  data_ultimo = data_ultimo[['ID', 'data_criacao', 'texto', 'num_favs', 'num_rts', 'Seguidores', 'hashtags', 'simbolos', 'urls', 'mencoes']]
  data_ultimo = data_ultimo.drop_duplicates('ID', inplace = True, keep = 'last')

  #atualizar o antigo
  data_ultimo = pd.concat([data_ultimo, lista_info_df])

  #remover o antigo
  remover_dataframe("geral_tweets_semanal")

  #subir o novo
  upload_dataframe(data_ultimo, nome = 'geral_tweets_semanal')


lista_ids_respondidos = []

def responde_tweet_cruzeiro(id):
  #tem que pegar os tweets e vrificar twwet por tweet
  global lista_ids_respondidos
  texto = 'Sou um bot que posta automaticamente números do engajamento do Cruzeiro no Twitter\n\nMe siga para estar por dentro dos números 🦊\n\n#FechadoComOCruzeiro\n@Cruzeiro'
  if id not in lista_ids_respondidos:
    api.update_status(texto, id)
    lista_ids_respondidos.append(id)

def extrai_tweets_busca():
  global ultimo_id

  palavras = []
  tweets = api.search_tweets(q='Cruzeiro', count = 100, since_id = ultimo_id)

  for i,tweet in enumerate(tweets):
    palavras.append([tweet.text, tweet.id])

  df_temp = pd.DataFrame(palavras, columns = ['Tweet', 'ID'])
  #ler o antigo
  df_palavras = ler_dataframe('Palavras')
  df_palavras = df_palavras[['Tweet', 'ID']]
  #atualizar o df
  df_palavras = pd.concat([df_palavras, df_temp])
  #apagar o antigo
  remover_dataframe('Palavras')
  #subir o novo
  upload_dataframe(df_palavras, nome = 'Palavras')

  ultimo_id = list(df_palavras.ID)[-1]


def gerar_nuvem():
  palavras = list(ler_dataframe('Palavras').Tweet)

  dia = datetime.datetime.now().strftime('%d/%m')

  palavras_cruzeiro = ' '.join([str(elem).upper() for elem in palavras])

  wordcloud = WordCloud(stopwords=stopwords,
                        background_color='white', width=1600,
                        height=800, max_words=50, mask=mascara, max_font_size=500, colormap = 'Blues',
                        min_font_size=5, contour_width=5, contour_color='#1e3d8f', random_state = 0).generate(palavras_cruzeiro)

  fig = plt.figure(frameon=False, dpi = 500)
  fig.set_size_inches(6,6.3)

  ax = plt.Axes(fig, [0., 0., 1., 1.])
  ax.set_axis_off()
  fig.add_axes(ax)

  ax.imshow(wordcloud, aspect='auto')
  fig.savefig("cruzeiro_wordcloud.jpg", dpi = 1000)

  api.update_status_with_media(f"Nuvem de Palavras do Cruzeiro {dia}\n\n#FechadoComOCruzeiro\n@Cruzeiro", "cruzeiro_wordcloud.jpg")

  remover_dataframe(nome = 'Palavras')
  df_temp = pd.DataFrame(columns = ['Tweet', 'ID'])
  upload_dataframe(df_temp, nome = 'Palavras')


  
def interacao_semana_grafico():
  data1 = datetime.datetime.now()-datetime.timedelta(days = 8)
  data2 = datetime.datetime.now()-datetime.timedelta(days = 1)

  api = autenticar()
  df = ler_dataframe("geral_tweets_semanal")
  df = df[['ID', 'data_criacao', 'texto', 'num_favs','num_rts', 'Seguidores', 'hashtags', 'simbolos', 'urls', 'mencoes']]
  df = df.drop_duplicates('ID', inplace = True, keep = 'last')
  df["interacoes"] = df.apply(lambda row: row.num_favs + row.num_rts, axis = 1)

  df['data_formatado'] = df['data_criacao'].apply(lambda x: converte_data(x))
  df = df[(df['data_formatado'] >= data1) & (df['data_formatado'] <= data2)]
  
  df = df.groupby(by = 'data_formatado').sum()

  cria_grafico(df = df, indicador = 0, nome = 'interacoes_semana.png')

  api.update_status_with_media(f"Evolução das interações na semana\n\n#FechadoComOCruzeiro\n@Cruzeiro", "interacoes_semana.png")



def interacao_mes_grafico():
  if datetime.datetime.now().day == 1:
    atualizacao_semana_limpo()

    mes_atual = datetime.datetime.now().month
    if mes_atual == 1:
      mes_referencia = 12
    else:
      mes_referencia = mes_atual - 1
    
    meses30 = [4, 6, 9, 11]
    meses31 = [1, 3, 5, 7, 8, 10, 12]
    meses28 = [2]

    if mes_referencia in meses30:
      numero_dias = 30
    elif mes_referencia in meses31:
      numero_dias = 31
    elif mes_referencia in meses28:
      numero_dias = 28
    else:
      numero_dias = 30

    data1 = datetime.datetime.now()-datetime.timedelta(days = numero_dias+1)
    data2 = datetime.datetime.now()-datetime.timedelta(days = 1)

    api = autenticar()

    df = ler_dataframe("geral_tweets_semanal")
    df = df[['ID', 'data_criacao', 'texto', 'num_favs', 'num_rts', 'Seguidores', 'hashtags', 'simbolos', 'urls', 'mencoes']]
    df = df.drop_duplicates('ID', inplace = True, keep = 'last')

    df["interacoes"] = df.apply(lambda row: row.num_favs + row.num_rts, axis = 1)

    df['data_formatado'] = df['data_criacao'].apply(lambda x: converte_data(x))
    df = df[(df['data_formatado'] >= data1) & (df['data_formatado'] <= data2)]
    
    df = df.groupby(by = 'data_formatado').sum()
    print(df['data_formatado'].sort_values())
    
    cria_grafico(df = df, indicador = 1, nome = 'interacoes_mes.png')

    api.update_status_with_media(f"Evolução das interações no mês", "interacoes_mes.png")


api = autenticar()

ultimo_id = 1481083132581453824

nltk.download('stopwords')
stopwords = nltk.corpus.stopwords.words('portuguese')
adicionaveis = ['t','cara','pode','vi','pro',"Tempo", 'vez', 'sim', 'não', 'nao','Like', 'React', 'Reply', 'More', 'vc', 'May', 'Like', 'Deu', 'https', 'co', 'https co', 'cruzeiro', 'vai', 'rt', 'RT', 'pra', 'demais', 'dorme', 'dia', 'hoje', 'agora']

remover_dataframe(nome = 'Palavras')
df_temp = pd.DataFrame(columns = ['Tweet', 'ID'])
upload_dataframe(df_temp, nome = 'Palavras')

for palavra in adicionaveis:
  stopwords.append(palavra)

mascara = np.array(Image.open("cincoestrela.PNG"))


seguidores_antigo = 2330295
tweets_antigo = 9
likes_antigo = 2602
rts_antigo = 188

tweets_antigo_dia = 98
likes_antigo_dia = 13596
rts_antigo_dia = 1157

tweets_antigo_semana = 102
likes_antigo_semana = 135859
rts_antigo_semana = 11935


scheduler1 = schedule.Scheduler()
scheduler2 = schedule.Scheduler()
scheduler3 = schedule.Scheduler()
scheduler4 = schedule.Scheduler()
scheduler5 = schedule.Scheduler()
scheduler6 = schedule.Scheduler()
scheduler7 = schedule.Scheduler()
scheduler8 = schedule.Scheduler()
scheduler9 = schedule.Scheduler()
scheduler10 = schedule.Scheduler()
scheduler11 = schedule.Scheduler()
scheduler12 = schedule.Scheduler()
scheduler13 = schedule.Scheduler()

scheduler1.every().day.at("23:40").do(atualizacao_dia, dia = datetime.datetime.now().replace(tzinfo=pytz.UTC)-datetime.timedelta(hours = 4))
scheduler2.every().monday.at("08:00").do(atualizacao_semana)
scheduler3.every(1).hours.at(":55").do(extrai_tweets_busca)
scheduler4.every().day.at("23:30").do(gerar_nuvem)
scheduler5.every().monday.at("09:00").do(interacao_semana_grafico)
scheduler6.every().day.at("10:00").do(interacao_mes_grafico)
scheduler7.every().day.at("08:15").do(atualizacao_hora)
scheduler8.every().day.at("11:15").do(atualizacao_hora)
scheduler9.every().day.at("14:15").do(atualizacao_hora)
scheduler10.every().day.at("17:15").do(atualizacao_hora)
scheduler11.every().day.at("20:15").do(atualizacao_hora)
scheduler12.every().day.at("23:15").do(atualizacao_hora)
scheduler13.every().day.at("02:15").do(atualizacao_hora, 6)

#atualizacao_hora(18)
#interacao_semana_grafico()
while True:
    # run_pending needs to be called on every scheduler
    scheduler1.run_pending()
    scheduler2.run_pending()
    scheduler3.run_pending()
    scheduler4.run_pending()
    scheduler5.run_pending()
    scheduler6.run_pending()
    scheduler7.run_pending()
    scheduler8.run_pending()
    scheduler9.run_pending()
    scheduler10.run_pending()
    scheduler11.run_pending()
    scheduler12.run_pending()
    scheduler13.run_pending()
    time.sleep(30)