from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, MaxPooling1D, Conv1D, GlobalMaxPooling1D, Dropout, LSTM, GRU
from tensorflow.keras import utils
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import utils
import requests
from bs4 import BeautifulSoup
import pandas
import re
import matplotlib.pyplot as plt
from time import sleep
import os
import urllib.request
import csv
import pickle


headers = {
            'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
            'accept-encoding': 'gzip, deflate, br',
            'accept-language': 'uk,en-US;q=0.9,en;q=0.8,ru;q=0.7',
            'sec-ch-ua': '"Google Chrome";v="89", "Chromium";v="89", ";Not A Brand";v="99"',
            'sec-ch-ua-mobile': '?0',
            'sec-fetch-dest': 'document',
            'sec-fetch-mode': 'navigate',
            'sec-fetch-site': 'none',
            'sec-fetch-user': '?1',
            'upgrade-insecure-requests': '1',
            'user-agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.90 Safari/537.36'
            }

def get_pages():
    for page_num in range(148):
        url = f'https://www.metacritic.com/browse/movies/score/metascore/all/filtered?sort=desc&page={page_num}'
        r = requests.get(url, headers=headers, allow_redirects=False)
        soup = BeautifulSoup(r.content, 'lxml')
        with open(f'.\pages\page{page_num}.txt', 'w', encoding = 'utf-8') as f:
            f.write(str(soup))

def parse():     
    df = pandas.DataFrame()
    places, titles, dates, ratings, sumarys, metascores, years, hrefs = [], [], [], [], [], [], [], []
    for page_num in range(148):
        with open(f'.\pages\page{page_num}.txt', 'r', encoding = 'utf-8') as f:
            soup = BeautifulSoup(f.read(), 'lxml')
            films_info = soup.find_all('td', class_='clamp-summary-wrap')
            for film in films_info:
                place = film.find('span', class_='title numbered').text
                title = film.find('a', class_='title').find('h3').text
                href = film.find('a', class_='title').get('href')
                clamp_details = film.find('div', class_='clamp-details').find_all('span')
                try:
                    date = clamp_details[0].text
                except:
                    date = '-'
                try:
                    rating = clamp_details[1].text
                except:
                    rating = 'Not Rated'
                sumary = film.find('div', class_='summary').text
                metascore = film.find('div', class_='clamp-score-wrap').find('a').text
                places.append(re.sub("^\s+|\n|\t|\r|\s+$", " ", place).replace('.', ''))
                titles.append(title)
                dates.append(date)
                ratings.append(rating.replace(' | ', ''))
                sumarys.append(re.sub("^\s+|\n|\t|\r|\s+$", " ", sumary))
                metascores.append(re.sub("^\s+|\n|\t|\r|\s+$", " ", metascore))
                years.append(date[-4:])
                hrefs.append(href)
    df['place'] = places
    df['title'] = titles
    df['date'] = dates
    df['rating'] = ratings
    df['sumary'] = sumarys
    df['metascore'] = metascores
    df['year'] = years
    df['url'] = hrefs
    df.to_csv('metacritic.csv', sep='|', index=False)
    

def download_critic_reviews():
    df=pandas.read_excel('metacritic.xlsx')
    for i, row in df.iterrows():
        url = f'https://www.metacritic.com{row["url"]}/critic-reviews'
        r = requests.get(url, headers=headers, allow_redirects=False)
        name = row['url'].replace('/movie/', '')
        soup = BeautifulSoup(r.content, 'lxml')
        if (str(soup) == ''):
            sleep(2)
            url = f'https://www.metacritic.com{row["url"]}-{row["year"]}/critic-reviews'
            r = requests.get(url, headers=headers, allow_redirects=False)
            soup = BeautifulSoup(r.content, 'lxml')
        with open(f'.\critic_pages\{name}.txt', 'w', encoding = 'utf-8') as f:
            f.write(str(soup))
        sleep(2)
        
def update_critic_reviews():
    files = os.listdir('critic_pages')
    df = pandas.read_excel('metacritic.xlsx')
    for f in files:
        if os.path.getsize(f'./critic_pages/{f}') < 2000 and f.endwith('.txt'):
            path = '/movie/'+f.replace('.txt','')
            loc_df = df.loc[df['url'] == path]
            url = f'https://www.metacritic.com{path}/critic-reviews'
            r = requests.get(url, headers=headers, allow_redirects=False)
            soup = BeautifulSoup(r.content, 'lxml')
            if (str(soup) == ''):
                sleep(2)
                url = f'https://www.metacritic.com{path}-{loc_df["year"].iloc[0]}/critic-reviews'
                r = requests.get(url, headers=headers, allow_redirects=False)
                soup = BeautifulSoup(r.content, 'lxml')
                print(f)
            with open(f'.\critic_pages\{f}', 'w', encoding = 'utf-8') as file:
                file.write(str(soup))
            sleep(2)

def update_404():
    files = os.listdir('critic_pages')
    df = pandas.read_excel('metacritic.xlsx')
    for f in files:
        if f.endwith('.txt'):
            with open(f'./critic_pages/{f}', encoding = 'utf-8') as read_f:
                soup = BeautifulSoup(read_f.read(), 'lxml')
                title = soup.find('title').text
                if title == '404 Page Not Found - Metacritic - Metacritic':
                    url = f'https://www.metacritic.com/movie/' + f.replace('.txt', '')
                    req = urllib.request.Request(url, data=None, headers=headers)
                    response = urllib.request.urlopen(req)
                    sleep(2)
                    print(f)
                    print(response.geturl())
                    r = requests.get(response.geturl()+'/critic-reviews',headers=headers)
                    sleep(2)
                    soup = BeautifulSoup(r.content, 'lxml')
                    with open(f'.\critic_pages\{f}', 'w', encoding = 'utf-8') as file:
                        file.write(str(soup))

def parse_reviews():
    df = pandas.read_excel('metacritic.xlsx')
    with open('reviews.csv', mode='w', encoding = 'utf-8') as review_file:
        writer =  csv.writer(review_file, delimiter='|', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['film_id', 'critic', 'review', 'score'])
        for i, row in df.iterrows():
                name = row['url'].replace('/movie/', '')
                with open(f'./critic_pages/{name}.txt', encoding = 'utf-8') as f:
                    soup = BeautifulSoup(f.read(), 'lxml')
                    reviews_info = soup.find_all('div', class_='review pad_top1 pad_btm1')
                    print(row['place'])
                    for review in reviews_info:
                        film_id = row['place']
                        score = review.find('div', class_ = 'left fl').find('div').text
                        try:
                            critic = review.find('span', class_ = 'author').text
                        except:
                            critic = '-'
                        try:
                            review_text = review.find('div', class_ = 'summary').find('a', class_ = 'no_hover').text
                        except:
                            review_text = review.find('div', class_ = 'summary').text
                        review_text = re.sub("^\s+|\n|\t|\r|\s+$", " ", review_text)
                        writer.writerow([film_id, critic, review_text, score])

def classes_reviews():
    df = pandas.read_csv('reviews.csv', sep='|')
    classes = []
    for i, row in df.iterrows():
        if row['score'] > 60:
            classes.append(1)
        elif row['score'] > 40:
            classes.append(2)
        else:
            classes.append(3)
    df['class'] = classes
    df.to_csv('reviews.csv', sep='|', index=False)

def neural_network():
    train = pandas.read_csv('reviews.csv', sep='|')
    reviews = train['review']
    num_words = 500000
    max_len = 180
    nb_classes = 3
    y_train = utils.to_categorical(train['class'] -1, nb_classes)
    tokenizer = Tokenizer(num_words=num_words)
    tokenizer.fit_on_texts(reviews)
    with open('tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    sequences = tokenizer.texts_to_sequences(reviews)
    x_train = pad_sequences(sequences, maxlen=max_len)
    cnn_model(x_train, y_train, max_len, num_words)
    lstm_model(x_train, y_train, max_len, num_words)
    gru_model(x_train, y_train, max_len, num_words)

def cnn_model(x_train, y_train, max_len, num_words):
    model_cnn = Sequential()
    model_cnn.add(Embedding(num_words, 32, input_length=max_len))
    model_cnn.add(Conv1D(250, 5, padding='valid', activation='relu'))
    model_cnn.add(GlobalMaxPooling1D())
    model_cnn.add(Dense(128, activation='relu'))
    model_cnn.add(Dense(3, activation='softmax'))
    model_cnn.compile(optimizer='adam', 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])
    model_cnn_save_path = 'best_model_cnn.h5'
    checkpoint_callback_cnn = ModelCheckpoint(model_cnn_save_path, 
                                      monitor='val_accuracy',
                                      save_best_only=True,
                                      verbose=1)
    history_cnn = model_cnn.fit(x_train, 
                            y_train, 
                            epochs=5,
                            batch_size=128,
                            validation_split=0.1,
                            callbacks=[checkpoint_callback_cnn])


def lstm_model(x_train, y_train, max_len, num_words):
    model_lstm = Sequential()
    model_lstm.add(Embedding(num_words, 32, input_length=max_len))
    model_lstm.add(LSTM(16))
    model_lstm.add(Dense(4, activation='softmax'))
    model_lstm.compile(optimizer='adam', 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    model_lstm_save_path = 'best_model_lstm.h5'
    
    checkpoint_callback_lstm = ModelCheckpoint(model_lstm_save_path, 
                                          monitor='val_accuracy',
                                          save_best_only=True,
                                          verbose=1)
    
    history_lstm = model_lstm.fit(x_train, 
                              y_train, 
                              epochs=5,
                              batch_size=128,
                              validation_split=0.1,
                              callbacks=[checkpoint_callback_lstm])

def gru_model(x_train, y_train, max_len, num_words):
    model_gru = Sequential()
    model_gru.add(Embedding(num_words, 32, input_length=max_len))
    model_gru.add(LSTM(16))
    model_gru.add(Dense(4, activation='softmax'))
    model_gru.compile(optimizer='adam', 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    model_gru_save_path = 'best_model_gru.h5'
    
    checkpoint_callback_gru = ModelCheckpoint(model_gru_save_path, 
                                          monitor='val_accuracy',
                                          save_best_only=True,
                                          verbose=1)
    
    history_gru = model_gru.fit(x_train, 
                              y_train, 
                              epochs=5,
                              batch_size=128,
                              validation_split=0.1,
                              callbacks=[checkpoint_callback_geu])
    
if __name__ == "__main__":
    get_pages()
    parse()
    download_critic_reviews()
    update_critic_reviews()
    update_404()
    classes_reviews()
    neural_network()
