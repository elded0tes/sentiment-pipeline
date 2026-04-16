import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import os


load_dotenv()

def init_driver():
    #Driver headless mode
    options = webdriver.ChromeOptions()

    options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('user-agent=Mozilla/5.0')

    driver = webdriver.Chrome(
        service=Service(ChromeDriverManager().install()),
        options=options
        )
    return driver

def login_glassdoor(driver):
    #login in glassdoor
    driver.get("https://www.glassdoor.com.mx/index.htm")
    time.sleep(2)

    email_field = driver.find_element(By.ID, 'inlineUserEmail')
    email_field.send_keys(os.getenv('GLASSDOOR_EMAIL'))
    driver.find_element(By.XPATH, "//button[@type='submit'").click()
    time.sleep()

    password_field = driver.find_element(By.ID, 'inlineUserPassword')
    password_field.send_keys(os.getenv("GLASSDOOR_PASSWORD"))
    driver.find_element(By.XPATH, "//button[@type='submit']").click()
    time.sleep(3)
    print('SUCCESSFULL LOGIN :)')
    
def scrape_reviews(company_url: str, max_pages: int = 5):
    #args: review page url, max scrapping pages
    #returns: DF with reviews
    driver = init_driver()
    login_glassdoor(driver)

    reviews = []

    for page in range(1, max_pages + 1):
        url = f'{company_url}?sort.sortType=RD&sort.ascending=false&filter.iso3Language=eng&page={page}'
        driver.get(url)
        time.sleep(3)

        #Close pop-ups 
        try:
            close_btn = driver.find_element(By.CLASS_NAME, 'modal_closeIcon')
            close_btn.click()
            time.sleep(1)
        except:
            pass
            
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        reviews_cards = soup.find_all('li', {'class':'empReview'})

        if not reviews_cards:
            print(f'ANY REVIEWS FOUND IN PAGE {page}')
            break

        for card in reviews_cards:
            try:
                rating = card.find('span', {'class':'gdStars'})
                rating = float(rating['title']) if rating else None

                title = card.find('a', {'class':'reviewLink'})
                title = title.text.strip() if title else ''

                pros = card.find('p', {'class':'pros'})
                pros = pros.text.strip() if pros else ''

                cons = card.find('p', {'class':'cons'})
                cons = cons.text.strip() if cons else''

                reviews.append({
                    'rating':rating,
                    'title':title,
                    'pros':pros,
                    'cons':cons,
                    'text':f'{pros}{cons}'.strip()
                })
            except Exception as e:
                print(f'Review error {e}')
                continue

        print(f'SCRAPED PAGE {page} - {len(reviews_cards)} REVIEWS')

    driver.quit()
    df = pd.DataFrame(reviews)
    df.to_csv('data/raw/reviews.csv', index=False)
    print(f'\n TOTAL EXTRACTED REVIEWS: {len(df)}')
    return df






    
    
                        
        
    
    
