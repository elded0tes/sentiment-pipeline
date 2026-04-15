import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by
import By
from selenium.webdriver.chrome.service
import Service
from selinium.webdriver.support.ui
import WebDriverWait
from selenium.webdriver.support
import expected_conditions as EC
from webdriver_manager.chrome
import ChromeDriverManager
from bs4 import BeatifulSoup
from dotenv import load_dotenv
import os

load_dotenv()

def init_driver():
    #Driver en modo headless
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
    
    
                        
        
    
    
