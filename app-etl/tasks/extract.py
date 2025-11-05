import re
import pandas as pd
import time
import requests
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
import json
from pathlib import Path
from datetime import datetime
# from airflow import DAG
# from airflow.operators.python import PythonOperator
from typing import Dict, Any
import os 
import sys
from dotenv import load_dotenv
from kafka import KafkaProducer
load_dotenv()

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / 'app-ml' / 'src'))
os.chdir(project_root) # Change directory to read the files from ./data folder

class ExtractTool:
    """
    An ExtractTool for scraping data from the real estate website, and save them in /data/raw
    """
    def __init__(self, config: Dict[str, Any]):
        """
        ExtractTool class with a configuration dictionary.
        
        Args:
            config: Dict[str, Any]: Configuration params for path and filename.
        Returns:
            None
        """
        self.api_key1 = os.getenv("LOCATIONIQ_KEY_1")
        self.api_key2 = os.getenv("LOCATIONIQ_KEY_2")
        self.api_key3 = os.getenv("LOCATIONIQ_KEY_3")
        self.api_key4 = os.getenv("LOCATIONIQ_KEY_4")

        self.config = config
        # self.producer = KafkaProducer(bootstrap_servers=['broker:29092'], max_block_ms=5000)
        self.url = "https://us1.locationiq.com/v1/search"
        self.producer = KafkaProducer(bootstrap_servers=['localhost:9092'], max_block_ms=5000)

    # def __call__(self, address, api_key):
    #     return self.extract(address, api_key)

    def extract_property(self):
        records = []
        headers = ["address", "bedroom_nums", "bathroom_nums", "car_spaces", "land_size", "price", "lat", "lon", "postcode", "city"]
        records.append(headers)

        cnt = 6500
        for step in range(2, 6):  # House size buckets
            driver = uc.Chrome()
            house_size_min = 200 + step * 200
            house_size_max = 200 + (step + 1) * 200
            print(f"Scraping size: {house_size_min}-{house_size_max} m²")

            for i in range(1, 80):  # Pages
                # url = f"https://www.realestate.com.au/sold/property-house-size-{house_size_min}-{house_size_max}-in-nsw/list-{i}?maxSoldAge=1-month&source=refinement"
                url = f"https://www.realestate.com.au/sold/property-house-size-{house_size_min}-{house_size_max}-in-nsw/list-{i}?source=refinement"
                driver.get(url)
                time.sleep(1)

                if cnt < 4900:
                    api_key = self.api_key1
                elif cnt < 9800:
                    api_key = self.api_key2
                elif cnt < 14700:
                    api_key = self.api_key3
                elif cnt < 19600:
                    api_key = self.api_key4
                else:
                    break

                property_cards = driver.find_elements(By.CLASS_NAME, "residential-card__content")
                if not property_cards:
                    print(f"No more listings on page {i}, moving to next size bucket.")
                    break

                for card in property_cards:
                    try:
                        address = card.find_element(By.CLASS_NAME, "residential-card__details-link").text.strip()
                    except:
                        address = None

                    bedroom_nums = bathroom_nums = car_spaces = land_size = price = None
                    features = card.find_elements(By.XPATH, ".//ul[contains(@class, 'residential-card__primary')]//li[@aria-label]")
                    for item in features:
                        label = item.get_attribute("aria-label").lower()
                        value = label.split(" ")[0]
                        if "bedroom" in label:
                            bedroom_nums = value
                        elif "bathroom" in label:
                            bathroom_nums = value
                        elif "car" in label:
                            car_spaces = value
                        elif "size" in label:
                            land_size = value

                    try:
                        price = card.find_element(By.CLASS_NAME, "property-price").text.strip()
                    except:
                        price = None
                    lat, lon, postcode, city = self.extract_address(address, api_key)
                    data = {
                        "address": address,
                        "bedroom_nums": bedroom_nums,
                        "bathroom_nums": bathroom_nums,
                        "car_spaces": car_spaces,
                        "land_size": land_size,
                        "price": price,
                        "lat": lat,
                        "lon": lon,
                        "postcode": postcode,
                        "city": city
                    }
                    # self.producer.send('house_price_scrape', json.dumps(data).encode('utf-8'))
                    print(f"Count: {cnt}", address, bedroom_nums, bathroom_nums, car_spaces, land_size, price, lat, lon, postcode, city)
                    records.append([address, bedroom_nums, bathroom_nums, car_spaces, land_size, price, lat, lon, postcode, city])
                    self.producer.send('house_scraping', json.dumps(data).encode('utf-8'))
                    cnt += 1
                time.sleep(1)
                # break
            driver.quit()
            # break
        df = pd.DataFrame(records[1:], columns=records[0])
        return df


    def extract_address(self, address, api_key):
        """
        Extracting steps:
        Get "address", "bedroom_nums", "bathroom_nums", "car_spaces", "land_size", "price" attributes from the website,
        Save raw file as .parquet format and .json for manifest file with date for better controlling
        Args:
            None
        Returns:
            None
        """
        lat, lon, postcode, city = None, None, None, None
        params = {
            'key': api_key,
            'q': address,
            'format': 'json'
        }
        try:
            response = requests.get(self.url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            if isinstance(data, list) and len(data) > 0:
                lat = float(data[0]['lat'])
                lon = float(data[0]['lon'])
                match = re.search(r'\b\d{4}\b', data[0]['display_name'])
                postcode = int(match.group()) if match else None
                display_parts = [part.strip() for part in data[0]['display_name'].split(',')]
                # print(display_parts)
                if 'Sydney' in display_parts:
                    city = 'Sydney'
                elif 'Newcastle' in display_parts:
                    city = 'Newcastle'
                elif 'Wollongong' in display_parts:
                    city = 'Wollongong'
                else:
                    city = None
            else:
                lat, lon, postcode, city = None, None, None, None
        except Exception as e:
            # print(f"Error on row {i}: {e}")
            lat, lon, postcode, city = None, None, None, None

        return lat, lon, postcode, city 
    
# import yaml
# config_path = r"D:\end_to_end_house_price_prediction\config\config.yaml" 
# with open(config_path, "r") as f: 
#     config = yaml.safe_load(f) 

# test = ExtractTool(config=config) 
# test.extract_property()