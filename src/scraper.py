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
    
