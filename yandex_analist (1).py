#!/usr/bin/env python
# coding: utf-8

# # Описание проекта

# <div style="border:solid blue 2px; padding: 20px"> 
#     
# 
# Вы работаете в интернет-магазине «Стримчик», который продаёт по всему миру компьютерные игры. Из открытых источников доступны исторические данные о продажах игр, оценки пользователей и экспертов, жанры и платформы (например, Xbox или PlayStation). Вам нужно выявить определяющие успешность игры закономерности. Это позволит сделать ставку на потенциально популярный продукт и спланировать рекламные кампании.
#     
#     
# Перед вами данные до 2016 года. Представим, что сейчас декабрь 2016 г., и вы планируете кампанию на 2017-й. Нужно отработать принцип работы с данными. Не важно, прогнозируете ли вы продажи на 2017 год по данным 2016-го или же 2027-й — по данным 2026 года.
#     
#     
# <b>Описание данных games.csv</b>
#     
# - Name — название игры
#     
# - Platform — платформа
#     
# - Year_of_Release — год выпуска
#     
# - Genre — жанр игры
#     
# - NA_sales — продажи в Северной Америке (миллионы долларов)
#     
# - EU_sales — продажи в Европе (миллионы долларов)
#     
# - JP_sales — продажи в Японии (миллионы долларов)
#     
# - Other_sales — продажи в других странах (миллионы долларов)
#     
# - Critic_Score — оценка критиков (от 0 до 100)
#     
# - User_Score — оценка пользователей (от 0 до 10)
#     
# - Rating — рейтинг от организации ESRB (англ. Entertainment Software Rating Board). Эта ассоциация определяет рейтинг компьютерных игр и присваивает им подходящую возрастную категорию.
#     
#     
# Данные за 2016 год могут быть неполными.
#     
# </div>

# # Шаг 2. Откройте файл с данными и изучите общую информацию

# In[1]:


import pandas as pd
import copy
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime # импортирую все необходимые библиотеки сразу
df = pd.read_csv('/games.csv')
df.info()


# In[2]:


df


# Проверяю данные в каждом столбце. Основательно изучаю столбцы, где есть пропуски.

# In[3]:


def check_columns(df):
    for column in df:
        print(column)
        print(df[column].unique())
check_columns(df)


# In[4]:


df[df['Name'].isna()]


# In[5]:


df[df['Year_of_Release'].isna()]


# In[6]:


df[df['Genre'].isna()]


# In[7]:


df[df['Critic_Score'].isna()]


# In[8]:


df[df['User_Score'].isna()]


# In[9]:


df[df['Rating'].isna()]


# # Выводы:

# <div style="border:solid blue 2px; padding: 20px"> 
#     
# Названия столбцов в датафрейме записаны в верхнем регистре.
# 
# Категориальные переменные: Name, Platform, Genre, Rating.
# Количественные переменные: Year_of_Release, NA_sales, EU_sales, JP_sales, Other_sales, Critic_Score, User_Score.
# 
# В данных присутствуют пропуски в столбцах: Name, Year_of_Release, Genre, Critic_Score, User_Score, Rating. Большое количество значений NaN в трех последних столбцах.
# 
# Тип данных был определен неверно в следующих столбцах: Year_of_release, User_score.
# 
# Также в столбце User_score были обнаружены некорректные значения 'tbd', а в столбцах Name, Genre и Year_of_Release были обнаружены дубликаты.
# 
# Данные требуют предварительной обработки.
# <\div>

# # Шаг 3. Подготовьте данные

# Привожу названия столбцов к нижнему регистру.

# In[10]:


df.columns = df.columns.str.lower()

print(df.head())


# Преобразую данные в нужные типы в столбцах Year_of_release (float64, а должен быть int64, т.к год - это целое число), User_score (object, а должен быть float64, т.к. оценка пользователей - это количественная переменная). 
# 
# Прежде, чем поменять тип в столбце User_score, необходимо убрать значения 'tbd'. Для этого я использую параметр 'errors ='coerce' метода to_numeric, который принудительно заменит некорректные значения на NaN. Это позволит не удалять строки и продолжить работу с Датафреймом

# In[11]:


df['user_score'] = pd.to_numeric(df['user_score'], errors='coerce')


# Данные в строках с NaN не стоит заменять на 0, т.к. это может привести к некорректным результатам статистического анализа. Удалять строки с пустыми данными во всех колонках, на мой взгляд, также нельзя. Данные отсутствуют в своем большистве всего лишь в 3 столбцах, а в 8 - они полноценные. Удалив такое кол-во строк, мы сильно исказим информацию, например, о данных по продажам или платформе.
# 
# Поэтому почистим строки только в 3х колонках с именем, жанром и годом релиза, т.к. строки без этих данных нам не пригодятся в анализе и восстановить их нельзя. Из 16715 строк удалим 271, т.е 2% от общего количества.

# In[12]:


df.dropna(subset=['name', 'genre', 'year_of_release'], inplace=True)
df['year_of_release'] = df['year_of_release'].astype(np.int64)
df.info()


# В столбце rating отсутствуют данные в 6 769 строках.
# ESRB часто повторяет один и тот же тип оценки для одинаковых жанров. Попробую восстановить данные в этом столбце, используя популярный ESRB рейтинг в жанре. Для этого необходимо определить самой популярную оценку для жанра и составить словарь.

# In[13]:


genre_list = df['genre'].unique()
genre_dict = {}
for genre in genre_list:
    print(genre)
    rating_value_counts = df[df['genre'] == genre]['rating'].value_counts()
    print(rating_value_counts)
    genre_dict[genre] = rating_value_counts.index[0]
    
genre_dict


# In[14]:


df['rating'] = df['rating'].fillna('NR')

def fill_rating(row):
    if row[10] == 'NR':
        row[10] = genre_dict[row[3]]
    return row

df = df.apply(fill_rating, axis=1)
df.info()


# In[15]:


df['rating'].value_counts()


# Посчитаю суммарные продажи во всех регионах и добавлю их в отдельный столбец total_sales.

# In[16]:


df['total_sales'] = df['na_sales'] + df['eu_sales'] + df['jp_sales'] + df['other_sales']
print(df.head())


# # Выводы:

# <div style="border:solid blue 2px; padding: 20px"> 
# Данные подготовлены к исследовательскому анализу:
# 
# 1) Названия столбцов приведены к нижнему регистру;
# 
# 2) Восстановлены пропуски в строках;
# 
# 3) Удалено минимальное кол-во строк, где отсутствовали необходимые данные для анализа;
# 
# 4) Добавлен столбец суммарных продаж.
# 
#     
# </div>

# # Шаг 4. Проведите исследовательский анализ данных

# <b> Вопрос 1.</b> *Посмотрите, сколько игр выпускалось в разные годы. Важны ли данные за все периоды?*

# In[17]:


df_by_year = df.pivot_table(index = 'year_of_release', values = 'name', aggfunc = 'count')
df_by_year.plot(kind='bar', figsize=(8, 8))


# <div style="border:solid blue 2px; padding: 20px"> 
# <b>Вывод:</b> 
# С 1980 по 1993 гг. было мало компаний и выпускалось мало игр. Объем выпуска игр начинает активно расти с 1994 года. Наибольшее количество игр было выпущено с 2008 по 2009 гг. Далее с 2010 года объем выпуска игр начал снижаться. 
# 
# Задача исследования "выявить определяющие успешность игры закономерности", поэтому для нас будут малоинформативны данные с 1980 по 1993 гг. Т.к. в тот период была низкая конкуренция, и игровая индустрия не была так развита.
#     
# </div>

# <b> Вопрос 2.</b> *Посмотрите, как менялись продажи по платформам. Выберите платформы с наибольшими суммарными продажами и постройте распределение по годам. Найдите популярные в прошлом платформы, у которых сейчас продажи на нуле. За какой характерный период появляются новые и исчезают старые платформы?*

# In[18]:


platform_grouped = df.pivot_table(index='platform', values='total_sales', aggfunc='sum').sort_values(
    by='total_sales', ascending=False)
platform_grouped = platform_grouped.head(6).reset_index()
platform_grouped.plot( x='platform', y='total_sales', kind='bar', figsize=(8, 9))
top_platforms_dict = platform_grouped['platform'].unique() #создадим словарь наиболее прибыльных платформ
platform_grouped.describe()


# In[19]:


#распределение продаж по годам для наиболее прибыльных платформ

for platform in top_platforms_dict:
    df[df['platform'] == platform].pivot_table(index='year_of_release', values='total_sales', aggfunc= 'sum').plot(kind='bar', figsize=(10,5))
    plt.title(platform)


# <div style="border:solid blue 2px; padding: 20px"> 
# <b>Вывод:</b> 
# Самые прибыльные по суммарным продажам оказались платформы: PS2, X360, PS3, Wii, DS, PS. У 
# этих же платформ к 2016 году практически нулевые продажи, т.к. их сменили новые поколения, например, платформы PS2 и PS3 сменила PS4. Из графиков видим, что срок жизни популярных игр в среднем составляет около 10 лет. 
#     
# </div>

# <b> Вопрос 3.</b> *Определите, данные за какой период нужно взять, чтобы исключить значимое искажение распределения по платформам в 2016 году.
# Далее работайте только с данными, которые вы определили. Не учитывайте данные за предыдущие годы.*

# In[20]:


actual_data = df.query('year_of_release > 2012')
actual_data.head().sort_values(by='year_of_release')


# <div style="border:solid blue 2px; padding: 20px"> 
# <b>Вывод:</b> Можно предположить, что в 2012 произошел кризис (почти в половину снизилось кол-во выпускаемых игр), или смена парадигмы в производстве игр. Упор сменился с объема выпускаемых игр на их качество. Поэтому актуальным периодом для дальнейшего анализа считаю 2013 - 2016 год.
#     
# </div>

# <b> Вопрос 4.</b> *Какие платформы лидируют по продажам, растут или падают? Выберите несколько потенциально прибыльных платформ.*

# In[21]:


platform_sales = actual_data.pivot_table(index='platform', values='total_sales', aggfunc='sum').sort_values(
    by='total_sales', ascending=False).plot(kind='bar', figsize=(8, 9))


# In[22]:


platform_sales_by_year = actual_data.pivot_table(index='year_of_release', columns = 'platform', values='total_sales',
                                        aggfunc='sum').plot(kind='bar', figsize=(10,10), style=dict)


# <div style="border:solid blue 2px; padding: 20px">
# <b>Вывод:</b>
# По продажам лидируют платформы: PS4, PS3, XOne, X360 и 3DS. Их суммарный доход за 3 года около 1 миллиарда долларов.
# 
# Продажи растут у PS4, XOne, 3DS, WiiU и PSV, так как они заменяют предыдущее поколение игровых приставок. На убыль идут приставки предыдущего поколения PS3, X360, DS, Wii и PSP.
#     
# </div>

# <b> Вопрос 5.</b> *Постройте график «ящик с усами» по глобальным продажам каждой игры и разбивкой по платформам. Велика ли разница в продажах? А в средних продажах на разных платформах? Опишите результат.*

# In[23]:


actual_data.boxplot(column = 'total_sales')
plt.ylim(0,1,25)
actual_data['total_sales'].describe()


# In[24]:


PS4_actual_data = actual_data.query('platform =="PS4"')
PS4_actual_data.boxplot(column = 'total_sales')
PS4_actual_data['total_sales'].describe()


# In[25]:


PS3_actual_data = actual_data.query('platform =="PS3"')
PS3_actual_data.boxplot(column = 'total_sales')
PS3_actual_data['total_sales'].describe()


# In[26]:


XOne_actual_data = actual_data.query('platform =="XOne"')
XOne_actual_data.boxplot(column = 'total_sales')
XOne_actual_data['total_sales'].describe()


# In[27]:


X360_actual_data = actual_data.query('platform =="X360"')
X360_actual_data.boxplot(column = 'total_sales')
X360_actual_data['total_sales'].describe()


# In[28]:


_3DS_actual_data = actual_data.query('platform =="3DS"')
_3DS_actual_data.boxplot(column = 'total_sales')
_3DS_actual_data['total_sales'].describe()


# <div style="border:solid blue 2px; padding: 20px"> 
# <b>Вывод:</b> 
# Разница в продажах существенная. Есть очень популярные игры, которые продаются долгое время и дают много прибыли. А есть много игр, которые не смогли преодолеть порог в 400 тыс. долларов за выбранный период.
# 
# Средние продажи на игру по миру: 488 тыс. дол. 3/4 игр в диапазоне до 400 тыс. Максимум 21 млн.
# 
# Средние продажи на игру по платформе PS4: 801 тыс. 3/4 игр в диапазоне до 730 тыс. Максимум 14,6 млн.
# 
# Средние продажи на игру по платформе PS3: 526 тыс. 3/4 игр в диапазоне до 510 тыс. Максимум 21 млн.
# 
# Средние продажи на игру по платформе XOne: 645 тыс. 3/4 игр в диапазоне до 685 тыс. Максимум 7,4 млн.
# 
# Средние продажи на игру по платформе X360: 735 тыс. 3/4 игр в диапазоне до 795 тыс. Максимум 16,3 млн.
# 
# Средние продажи на игру по платформе 3DS: 472 тыс. 3/4 игр в диапазоне до 280 тыс. Максимум 14,6 млн.
# 
# Положительная тенденция к росту продаж у платформ PS4, PS3 и 3DS. Они имеют средние продажи больше 3го квантиля, это говорит о том, что у этих платформ более популярные и продаваемые игры, чем на XOne и X360.
# </div>

# <b> Вопрос 6.</b> *Посмотрите, как влияют на продажи внутри одной популярной платформы отзывы пользователей и критиков. Постройте диаграмму рассеяния и посчитайте корреляцию между отзывами и продажами. Сформулируйте выводы и соотнесите их с продажами игр на других платформах.*

# In[29]:


PS4_sales_crit_and_user_ratings = PS4_actual_data.loc[:,['total_sales', 'critic_score', 'user_score']]
PS4_sales_crit_and_user_ratings.head()


# In[30]:


pd.plotting.scatter_matrix(PS4_sales_crit_and_user_ratings, figsize=(12, 12))


# In[31]:


PS4_sales_crit_and_user_ratings.corr()


# In[32]:


PS3_sales_crit_and_user_ratings = PS3_actual_data.loc[:,['total_sales', 'critic_score', 'user_score']]
PS3_sales_crit_and_user_ratings.head()


# In[33]:


pd.plotting.scatter_matrix(PS3_sales_crit_and_user_ratings, figsize=(12, 12))


# In[34]:


PS3_sales_crit_and_user_ratings.corr()


# In[35]:


XOne_sales_crit_and_user_ratings = XOne_actual_data.loc[:,['total_sales', 'critic_score', 'user_score']]
XOne_sales_crit_and_user_ratings.head()


# In[36]:


pd.plotting.scatter_matrix(XOne_sales_crit_and_user_ratings, figsize=(12, 12))


# In[37]:


XOne_sales_crit_and_user_ratings.corr()


# In[38]:


_3DS_sales_crit_and_user_ratings = _3DS_actual_data.loc[:,['total_sales', 'critic_score', 'user_score']]
_3DS_sales_crit_and_user_ratings.head()


# In[39]:


pd.plotting.scatter_matrix(_3DS_sales_crit_and_user_ratings, figsize=(12, 12))


# In[40]:


_3DS_sales_crit_and_user_ratings.corr()


# In[41]:


X360_sales_crit_and_user_ratings = X360_actual_data.loc[:,['total_sales', 'critic_score', 'user_score']]
X360_sales_crit_and_user_ratings.head()


# In[42]:


pd.plotting.scatter_matrix(X360_sales_crit_and_user_ratings, figsize=(12, 12))


# In[43]:


X360_sales_crit_and_user_ratings.corr()


# <div style="border:solid blue 2px; padding: 20px"> 
# <b>Вывод:</b>
# Для платформы PS4 есть прямая корреляция между продажами и рейтингом критиков. Можно предположить, что важным критерием для платформы является качество игры. 
# 
# Между рейтингом критиков и пользователей еще более сильная корреляция. Исходя из анализа, видим, что рейтинг игры влияет на мнение игроков. То есть, чем выше ценит игру критик, тем выше оценит ее игрок.
# 
# Также есть незначительная обратная корреляция между продажами и рейтингом пользователей, значит эти величины не зависят друг от друга. 
# 
# Платформы PS3, XOne и X360 имеют похожие связи, как и у PS4.
# 
# Можно выделить платфому 3DS, чьи продажи зависят напрямую, хоть и не сильно от рейтинга игроков. Можно предположить, что в данном направлении компания лучше ведет работу по составлению рейтинга для игр и системы отзывов для игроков.
# 
#    
# </div>

# <b> Вопрос 7.</b> *Посмотрите на общее распределение игр по жанрам. Что можно сказать о самых прибыльных жанрах? Выделяются ли жанры с высокими и низкими продажами?*

# In[44]:


genre_pvt = actual_data.pivot_table(index='genre', values='total_sales', aggfunc='sum')
genre_pvt.sort_values('total_sales', ascending=False).plot(kind='bar', figsize=(10,10))


# <div style="border:solid blue 2px; padding: 20px"> 
# <b>Вывод:</b>
# Самые прибыльные жанры: Action, Shooter, Sports и Role-Playing.
# 
# Игрокам больше нравятся активные жанры, основаннные на разнообразии действий ("Стрелялки" и "Боевики"), а также ролевые игры, где участвует много игроков. 
# 
# Игры логических жанров в сравнении с играми жанра action практически вообще не пользуются спросом.
# </div>

# # Шаг 5. Составьте портрет пользователя каждого региона

# <b> Вопрос 1.</b> 
# *Определите для пользователя каждого региона (NA, EU, JP):*
# 
# *- Самые популярные платформы (топ-5). Опишите различия в долях продаж.*

# In[45]:


top_5_na_sales = actual_data.query('na_sales > 0').groupby(['platform'], 
            as_index = False)['na_sales'].sum().sort_values('na_sales', 
                                ascending = False).head(5)['platform'].tolist()
 
for name in top_5_na_sales:
    actual_data.query('platform == @name').pivot_table(index = 'year_of_release',
                        values = ['na_sales'], aggfunc = 'sum').sort_values('year_of_release', 
                                                    ascending = False).plot(kind='bar',figsize = (10, 5), title = name)
    
    plt.xlabel('Дата релиза', labelpad = 10)
    plt.ylabel('Продажи', labelpad = 50)
    plt.legend()


# In[46]:


top_5_eu_sales = actual_data.query('eu_sales > 0').groupby(['platform'], 
            as_index = False)['eu_sales'].sum().sort_values('eu_sales', 
                                ascending = False).head(5)['platform'].tolist()
 
for name in top_5_eu_sales:
    actual_data.query('platform == @name').pivot_table(index = 'year_of_release',
                        values = ['eu_sales'], aggfunc = 'sum').sort_values('year_of_release', 
                                                    ascending = False).plot(kind='bar',figsize = (10, 5), title = name)
    
    plt.xlabel('Дата релиза', labelpad = 10)
    plt.ylabel('Продажи', labelpad = 50)
    plt.legend()


# In[47]:


top_5_jp_sales = actual_data.query('jp_sales > 0').groupby(['platform'], 
            as_index = False)['jp_sales'].sum().sort_values('jp_sales', 
                                ascending = False).head(5)['platform'].tolist()
 
for name in top_5_jp_sales:
    actual_data.query('platform == @name').pivot_table(index = 'year_of_release',
                        values = ['jp_sales'], aggfunc = 'sum').sort_values('year_of_release', 
                                                    ascending = False).plot(kind='bar',figsize = (10, 5), title = name)
    
    plt.xlabel('Дата релиза', labelpad = 10)
    plt.ylabel('Продажи', labelpad = 50)
    plt.legend()


# <div style="border:solid blue 2px; padding: 20px"> 
# <b>Вывод:</b>
# Рейтинг платформ по суммарным продажам в регионе:
# 
# Топ 5 по Северной Америке: PS4, XOne, X360, PS3 и 3DS
# 
# Топ 5 по Европейскому региону: PS4, PS3, XOne, X360 и 3DS
# 
# Топ 5 по Японии: 3DS, PS3, PSV, PS4 и WiiU
# 
# Получается, что за выбранный период 2013-2016 гг. по суммарным продажам в регионах есть незначительные различия в предпочтениях пользователей в Америке и Европе. Самые высокие продажи в этих регионах у PS4, платформа XOne по продажам на 2м месте. 
# 
# А в Японии PS4 находится на 4м месте, самые высокие суммарные продажи у 3DS (Нинтендо).
# 
# Но с разбивкой продаж по годам видна одна общая тенденция по всем регионам, PS4 охватывает все большую долю рынка и ее продажи растут.  У PS3 самые высокие продажи в 2013 году во всех регионах и к 2016  идет активный спад. 
# 
# В Америке и Европе к 2016 году продажи растут у XOne,  а вот в Японии на 2м месте PSV.
# </div>

# <b> Вопрос 2.</b> *- Самые популярные жанры (топ-5). Поясните разницу.*

# In[48]:


genre_by_area = actual_data.query('jp_sales > 0 and eu_sales > 0 and na_sales > 0').pivot_table(index='genre', 
    values=['na_sales','eu_sales','jp_sales'], aggfunc='sum').sort_values(by='genre')
print(genre_by_area)
genre_by_area.plot(kind='bar', figsize=(12, 12))


# <div style="border:solid blue 2px; padding: 20px"> 
# <b>Вывод:</b>
# Рейтинг жанров:
# 
# Топ 5 по Северной Америке: Action, Shooter, Sports, Role-Playing и Misc
# 
# Топ 5 по Европейскому региону: Action, Shooter, Sports, Role-Playing и Racing
# 
# Топ 5 по Японии: Role-Playing, Action, Misc, Fighting и Shooter
# 
# Как и в выборе платформы, так и в выборе по жанру Америка и Европа солидарны. Япония больше предпочитает ролевые игры, а потом уже экшн. В целом можно отметить, что японские игроманы выбирают более "вдумчивые" жанры.
#  </div>

# <b> Вопрос 2.</b> *Влияет ли рейтинг ESRB на продажи в отдельном регионе?*

# In[49]:


rating_by_area = actual_data.pivot_table(index='rating', values=['na_sales','eu_sales','jp_sales'], 
                aggfunc='sum').sort_values(by='rating')
print(rating_by_area)
rating_by_area.plot(kind='bar', figsize=(12, 12))


# <div style="border:solid blue 2px; padding: 20px"> 
# <b>Вывод:</b> Рейтинг возраста влияет по каждому региону.
# 
# В Америке и Европе схожие ситуации, лучше всего продаются игры с возрастным рейтингом М (17+). Следом идут игры для любого возраста. В Японии же лидируют по продажам игры с рейтингом 13+, также на 2м месте расположились игры с рейтингом Е (для всех возрастов).
# 
# </div>

# # Шаг 6. Проведите исследование статистических показателей

# *Как изменяется пользовательский рейтинг и рейтинг критиков в различных жанрах? Посчитайте среднее количество, дисперсию и стандартное отклонение. Опишите распределения.*

# In[50]:


genre_critic_score = actual_data.query('critic_score > 0').groupby(['genre'], 
            as_index = False)['critic_score'].sum().sort_values('critic_score', 
                                ascending = False).head(10)['genre'].tolist()
 
for name in genre_critic_score:
    actual_data.query('genre == @name').pivot_table(index = 'year_of_release',
                        values = ['critic_score'], aggfunc = ['sum', 'count','mean']).sort_values('year_of_release', 
                                                    ascending = False).plot(kind='bar',figsize = (10, 5), title = name)
    
    plt.xlabel('Год', labelpad = 10)
    plt.ylabel('Рейтинг критиков', labelpad = 50)
    plt.legend()


# In[51]:


genre_user_score = actual_data.query('user_score > 0').groupby(['genre'], 
            as_index = False)['user_score'].sum().sort_values('user_score', 
                                ascending = False).head(10)['genre'].tolist()
 
for name in genre_critic_score:
    actual_data.query('genre == @name').pivot_table(index = 'year_of_release',
                        values = ['user_score'], aggfunc = ['sum', 'count','mean']).sort_values('year_of_release', 
                                                    ascending = False).plot(kind='bar',figsize = (10, 5), title = name)
    
    plt.xlabel('Год', labelpad = 10)
    plt.ylabel('Рейтинг пользователей', labelpad = 50)
    plt.legend()


# In[52]:


genre_score = actual_data.pivot_table(index='genre', values=['critic_score','user_score'], 
                aggfunc=['sum', 'count', 'mean']).sort_values(by='genre')
genre_score


# In[53]:


actual_data.describe()


# In[54]:


actual_data['critic_score'].hist()


# In[55]:


actual_data['user_score'].hist()


# In[56]:


variance_estimate = np.var(actual_data['user_score'])
print(variance_estimate) 


# In[57]:


variance_estimate = np.var(actual_data['critic_score'])
print(variance_estimate) 


# <div style="border:solid blue 2px; padding: 20px">
# <b>Вывод:</b>
# Больше всего оценок, как у критиков, так и у пользователей по играм в наиболее популярных жанрах: Action, Shooter, Sports, Role-Playing. Средняя и медиана практически не отличаются, значит большого разброса по оценкам критиков и пользователей нет. В основном игры оцениваются положительно, по гистограммам видно, что критики чаще ставят оценки в диапазоне от 70 до 80. А пользователи - от 7 до 8. Но в общем сложилось впечатление, что пользователи немного благосклоннее.
#     
# </div>

# # Шаг 7. Проверьте гипотезы

# - *Средние пользовательские рейтинги платформ Xbox One и PC одинаковые;*
# 
# - *Средние пользовательские рейтинги жанров Action (англ. «действие») и Sports (англ. «виды спорта») разные.*
# 
# *Задайте самостоятельно пороговое значение alpha.*

# In[58]:


def stat_info(serie, bins=0): #Подготовка данных по первой гипотезе. Исключим игры с пустым рейтингом.
    serie_description = serie.describe()
    mean = serie_description[1]
    std = serie_description[2]
    d_min = serie_description[3]
    q1 = serie_description[4]
    median = serie_description[5]
    q3 = serie_description[6]
    d_max = serie_description[7]
    left_border = d_min
    right_border = d_max
    if bins == 0:
        bins = right_border - left_border
        if bins>100:
            bins = 100
        elif bins < 1:
            bins = abs(bins*10)+1
        bins = int(bins)
    else:
        bins = bins
    serie.hist(bins=bins, range=(left_border, right_border))
    print(serie_description)
    variance_estimate = np.var(serie, ddof=1)
    standart_dev = np.std(serie, ddof=1)
    print('Среднее значение: {:.2f}'.format(mean))
    print('Дисперсия: {:.2f}'.format(variance_estimate))
    print('Стандартное отклонение: {:.2f}'.format(standart_dev))
    return [mean, variance_estimate, standart_dev]


# In[59]:


user_rating_XOne = actual_data.query('platform == "XOne"')['user_score'].dropna()
stat_info(user_rating_XOne)


# In[60]:


user_rating_PC = actual_data.query('platform == "PC"')['user_score'].dropna()
stat_info(user_rating_PC)


# <div style="border:solid blue 2px; padding: 20px">
# Средние кол-ва значений похожи, дисперсия выборок отличается.
# 
# Нулевая гипотеза: "Средние пользовательские рейтинги платформ Xbox One и PC одинаковые".
# 
# Альтернативная гипотеза гласит: "Средние пользовательские рейтинги платформ Xbox One и PC различаются".
# 
# Для оценки гипотезы необходимо применить тест Стьюдента. 
# При получении ответа "Отвергаем нулевую гипотезу", делаю вывод, что данные различаются и нулевая гипотеза неверна.
# 
# При получении ответа "Не получилось отвергнуть нулевую гипотезу"- это значит, что нулевая гипотеза подтвердиась.
# </div>

# In[61]:



alpha = 0.01

results = st.ttest_ind(user_rating_XOne, user_rating_PC)

print('p-значение:', results.pvalue)

if (results.pvalue < alpha):
    print("Отвергаем нулевую гипотезу")
else:
    print("Не получилось отвергнуть нулевую гипотезу")


# Подготовим данные для жанров Action и Sports.

# In[62]:


user_rating_Action = actual_data.query('genre == "Action"')['user_score'].dropna()
stat_info(user_rating_Action)


# In[63]:


user_rating_Sports = actual_data.query('genre == "Sports"')['user_score'].dropna()
stat_info(user_rating_Sports)


# Нулевая гипотеза: "Средние пользовательские рейтинги жанров Action и Sports одинаковые".
# 
# Альтернативная гипотеза: "Средние пользовательские рейтинги жанров Action и Sports различаются".
# 
# Для оценки гипотезы также необходимо применить тест Стьюдента. 

# In[64]:


alpha = 0.01

results = st.ttest_ind(user_rating_Action, user_rating_Sports)

print('p-значение:', results.pvalue)

if (results.pvalue < alpha):
    print("Отвергаем нулевую гипотезу")
else:
    print("Не получилось отвергнуть нулевую гипотезу")


# <div style="border:solid blue 2px; padding: 20px">
# <b>Вывод:</b>
# Нулевая гипотеза об одинаковых средних пользовательских рейтингах платформ PC и XOne подтвердилась, это можно было наблюдать при анализе зависимости платформ, рейтинга и суммарных продаж.  Не зря данные платформы возглавляли топ по продажам, пользователи любят данные платформы и оценивают примерно одинаково. 
# Нулевая гипотеза о том, что средние пользовательские рейтинги жанров Action и Sports одинаковые была отвергнута. Жанр Action самый популярный среди игроков, в этом жанре выпускается большее кол-во игр. А также из проведенного статистического анализа мы знаем, что средний пользовательский рейтинг жанра Action = 6.837532, а жанра Sports = 5.238125.
#     
# </div>

# # Общий вывод 

# <div style="border:solid red 2px; padding: 20px">
# В ходе исследования было выполнено:
# 
# 1) Были изучены данные из файла, выявлены аномалии и пропущенные значения;
# 
# 2) Проведена подготовка данных (предобработка);
# 
# 3) Проведен исследовательский анализ данных, который на каждом этапе был подкреплен выводами;
# 
# 4) Составлен портрет пользователей каждого региона;
# 
# 5) Проведено исследование статистических показателей; 
# 
# 6) Выполнена проверка гипотез.
# 
# <b>Цель исследования - выявить определяющие успешность игры закономерности.</b>
# 
# В ходе анализа были выявлены тенденции развития игровой индустрии, с 2013 года сменилась парадигма выпуска игр с кол-ва на качество. Средняя продолжительность жизни популярной платформы около 10 лет. 
# 
# Лидерами среди игровых платформ являются PS4, ХОne для Америки и Европы, а для Японии PS4, PSV и WiiU.
# 
# Игровой портрет американских и европейких пользователей отличается от портрета японцев.
# 
# Американцы и европейцы выбирают платформы для игр PS4 и XONE, преобладающая целевая аудитория - это люди в возрасте 17+, которые в основном предпочитают игры  в жанрах: Action, Shooter, Sports, Role-Playing.
# 
# Японские пользователи выбирают платформы PS4 и PSМ (тенденция на 2016 год), преобладающая целевая аудитория - это подростки 13+, которые в основном предпочитают игры  в жанрах: Role-Playing, Action, Misc, Fighting и Shooter.
# 
# Стоит отметить, что в японском регионе отдают предпочтение своим производителям. В Америке и Европе наряду с японским лидером рынка PS4, также популярна "местная" платфома XONE, которая производится компанией Microsoft. 
# 
# При прогнозировании затрат на рекламные кампании обязательно нужно учитывать разные целевые аудитории по регионам. 
# 
# 
# </div>
