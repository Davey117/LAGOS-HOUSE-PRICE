import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

def scrape_jiji_houses(num_pages=5):
    # 1. Setup Chrome Options (Headless to run in background)
    chrome_options = Options()
    # chrome_options.add_argument("--headless")  # Uncomment to run without opening browser window
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--window-size=1920,1080")
    
    # fake a user agent so Jiji thinks we are a real person
    chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36")

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
    
    scraped_data = []
    
    print(f"Starting scrape for {num_pages} pages...")

    for page in range(1, num_pages + 1):
        url = f"https://jiji.ng/lagos/houses-apartments-for-sale?page={page}"
        print(f"Scraping Page {page}: {url}")
        
        driver.get(url)
        
        # 2. Wait for the listings to load (Robustness)
        try:
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CLASS_NAME, "b-list-advert__template"))
            )
        except:
            print(f"Timeout on page {page}. Skipping...")
            continue
        
        # 3. Find all listing cards on the page
        listings = driver.find_elements(By.CLASS_NAME, "b-list-advert__template")
        
        for card in listings:
            try:
                # Extracting specific details using relative XPaths
                # Note: These class names are current as of late 2025 but can change. 
                # Inspect Jiji's HTML if you get empty results.
                
                title = card.find_element(By.CLASS_NAME, "qa-advert-title").text
                price = card.find_element(By.CLASS_NAME, "qa-advert-price").text
                location = card.find_element(By.CLASS_NAME, "b-list-advert__region").text
                link = card.find_element(By.TAG_NAME, "a").get_attribute("href")
                
                # Jiji often puts the description in a sub-div
                try:
                    desc = card.find_element(By.CLASS_NAME, "b-list-advert__item-attr").text
                except:
                    desc = "N/A"

                scraped_data.append({
                    "Title": title,
                    "Price": price,
                    "Location": location,
                    "Description": desc,
                    "URL": link
                })
            except Exception as e:
                # Skip broken listings (Robustness)
                continue
        
        # Be nice to the server
        time.sleep(3) 

    driver.quit()
    
    # 4. Save to CSV
    df = pd.DataFrame(scraped_data)
    df.to_csv("jiji_lagos_houses.csv", index=False)
    print(f"Successfully saved {len(df)} listings to jiji_lagos_houses.csv")
    return df

# Run the function
if __name__ == "__main__":
    scrape_jiji_houses(pages=3)