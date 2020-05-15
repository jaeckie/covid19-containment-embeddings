# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 15:37:43 2020

@author: moder
"""
import os
from datetime import datetime
import pandas as pd
import urllib.request
from bs4 import BeautifulSoup 

user_agent = "user_agent = 'Mozilla/5.0 (Windows NT 6.1; Win64; x64)"

def scrap_wikipedia_text(url):
    request = urllib.request.Request(url, data=None, headers={'User-Agent' : user_agent})
    html = urllib.request.urlopen(request).read().decode('utf-8')   
    soup = BeautifulSoup(html, 'html.parser')
    content_div = soup.find('div', attrs={'id': 'mw-content-text'})
    # remove tables and graphs
    if content_div is not None: 
        for s in content_div.select('table'): 
            s.extract()
        for s in content_div.select('img'): 
            s.extract()
        # remove references
        for s in content_div.select('div.reflist'):
            s.extract()
            print('div.reflist extracted from %s...' % url)
        # iterate all p tags and append to text
        tags = ['h1', 'h2', 'h3', 'li', 'p']
        bodytext = ''
        for con in content_div.find_all(tags):
            bodytext += con.text      
        return bodytext 
    return None

if __name__ == '__main__':
    print('store data started...')  
    # load containment history file from kaggle
    df_contain = pd.read_csv(r'data/COVID 19 Containment measures data.csv')
    
    # cfilter = df_contain['Country'].isin(['Austria', 'Germany', 'Italy', 'Spain', 'Denmark'])
    # df_c = df_contain[cfilter]
    df_c = df_contain
    
    df =  df_c[df_c['Source'].notna()]
    df_drop = df.drop_duplicates(subset='Source', keep='last')
    
    wfilter = df_drop['Source'].str.contains('en.wikipedia.org')
    df_red = df_drop[wfilter]
       
    df_res = df_red[['Date Start', 'Country', 'Keywords', 'Source']]
    df_res.to_csv(r'data/covid19-all-countries.csv')

    for index, row in df_res.iterrows():
        text = scrap_wikipedia_text(row['Source'])
        time = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = '%s_%s_covid19-wikipedia.txt' % (time, row['Country'])
        with open(os.path.join('data',filename), 'w', encoding='utf-8') as file:
            file.write(text)
            print('saved file %s ...' % filename)
        file.close()
    # \[\d+\]
    
