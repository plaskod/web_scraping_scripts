from bs4 import BeautifulSoup
import requests

import lxml
import cchardet

import csv
import pandas as pd
from tqdm import tqdm

import numpy as np
from transformers import pipeline

URL = 'https://amazon.com/'
MAX_NUMBER_OF_PAGES = 5

def get_url(search_string):
  base_url = 'https://www.amazon.com/s?k={}&ref=nb_sb_noss_1'
  search_string = search_string.replace(' ', '+')
  url = base_url.format(search_string)
  # add page number 
  url += '&page{}'

  return url


def extract_data_from_amazon_item(item):
  """
    Extracts information from webpage such as:
    item's description,
    price,
    rating
    url

    @param item: html div as str of a specific item on the amazon webpage 
    @returns tuple of item's description, price, rating and url
  """
  atag = item.h2.a
  description = atag.text.strip()
  url = URL + atag.get('href')

  try:
    price_parent = item.find('span', 'a-price')
    price = price_parent.find('span', 'a-offscreen').text.replace(u'\xa0', ' ')
  except AttributeError:
    return

  try:
    rating = item.i.text
  except AttributeError:
    rating = ''

  return (description, price, rating,  url)


def scrape_amazon_for_item_details(search_terms, save_result_path, user_agent):
    """
        Scrape amazon webpage by finidng items, extracting their data and saving it as a csv file
        Relative path and browser's metadata such as user agent is required, because amazon blocks automated search
    """
    records = []
    for search_term in search_terms:
        url = get_url(search_term)
        for page_num in tqdm(range(1,MAX_NUMBER_OF_PAGES)):
            page = requests.get(url.format(page_num), headers=user_agent)
            soup = BeautifulSoup(page.content, 'lxml')
            results = soup.find_all('div', {'data-component-type': 's-search-result'})

            for item in results:
                record = extract_data_from_amazon_item(item)
                if record:
                    records.append(record)

    with open(save_result_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['description', 'price', 'rating', 'url'])
        for record in records:
            writer.writerow(record)


def get_reviews_from_page(soup):
  """
    Extract reviews (product name, title of the review, rating given by the reviewer and body of the review) from a given soup object (BeautifulSoup)
    @return a list of json
  """
  all_reviews = []
  reviews = soup.find_all('div', {'data-hook': 'review'})
  try:
    for item in reviews:
      review = {
          'product': soup.title.text.strip().replace('Amazon.plRecenzje klientów: ', ''),
          'title': item.find('span', {'data-hook': 'review-title'}).text.strip(),
          'rating': float(item.find('i', {'data-hook' : 'cmps-review-star-rating'}).text.strip().replace(' z 5 gwiazdek', '').replace(',', '.')),
          'body': item.find('span', {'data-hook': 'review-body'}).text.strip()
        }
      all_reviews.append(review)
  except:
    pass

  return all_reviews


def scrape_reviews(url, user_agent):
  """
    Iterates over page numbers and extracts reviews with: get_reviews_from_page()
    @return a list of json
  """
  reviews = []
  for page_num in tqdm(range(1, 10)):
    page = requests.get(f"{url}{page_num}", headers=user_agent)
    soup = BeautifulSoup(page.text, 'lxml')
    rvs = get_reviews_from_page(soup)
    # check if any review exists on the current page
    if not soup.find('li', {'class':'a-disabled a-last'}):
      pass
    else:
      break
    
    if len(rvs) != 0: 
      reviews.append(rvs)

  return reviews



def _test():
    """
      Test to extract product information for a given search_term 
    """
    
    search_terms = ['nvidia gpu']
    user_agent = {
      'Host': 'www.amazon.pl',
      'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/103.0.0.0 Safari/537.36',
      'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
      'Accept-Language': 'en-US,en;q=0.9',
      'Accept-Encoding': 'gzip, deflate, br',
      'Connection': 'keep-alive',
      'Upgrade-Insecure-Requests': '1',
      'TE': 'Trailers'
    }

    scrape_amazon_for_item_details(search_terms, './amazon_gpu.csv', user_agent)
    df = pd.read_csv('./amazon_gpu.csv')
    print(df.shape)

def _test2():
  """
    Test to extract reviews from a given amazon url and perform sentiment analysis using roBERTa-large
  """

  url = 'https://www.amazon.com/PNY-QUADRO-5000-Graphic-Card/product-reviews/B07JG1YBKY/ref=cm_cr_dp_d_show_all_btm?ie=UTF8&reviewerType=all_reviews&pageNumber='
  user_agent = {
      'Host': 'www.amazon.pl',
      'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/103.0.0.0 Safari/537.36',
      'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
      'Accept-Language': 'en-US,en;q=0.9',
      'Accept-Encoding': 'gzip, deflate, br',
      'Connection': 'keep-alive',
      'Upgrade-Insecure-Requests': '1',
      'TE': 'Trailers'
    }
    
  results = scrape_reviews(url, user_agent)
  reviews = np.array(results).flatten()

  sentiment_analysis = pipeline("sentiment-analysis", model="siebert/sentiment-roberta-large-english")
  
  for review in reviews:
    print(review['body'], sentiment_analysis(review['body']))

if __name__ == '__main__':
    _test2()