import os
import time
import urllib.request
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys

# Категории и их поисковые запросы
CATEGORIES = {
    "stalker": "S.T.A.L.K.E.R. game screenshot",
    "minecraft": "Minecraft game screenshot",
    "real_photo": "Real-world photo"
}

OUTPUT_DIR = "dataset"
NUM_IMAGES = 51  # Количество изображений на класс (уменьшил для теста)

# Запуск Chrome WebDriver
options = webdriver.ChromeOptions()
driver = webdriver.Chrome(options=options)

def download_images(category, query):
    os.makedirs(os.path.join(OUTPUT_DIR, category), exist_ok=True)
    driver.get("https://www.google.com/imghp")
    time.sleep(2)

    search_box = driver.find_element(By.NAME, "q")
    search_box.send_keys(query)
    search_box.send_keys(Keys.RETURN)
    time.sleep(3)

    for _ in range(3):  # Прокрутка страницы вниз для загрузки новых изображений
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(2)

    images = driver.find_elements(By.TAG_NAME, "img")
    count = 0
    print(f"[INFO] Начинаем загрузку изображений для категории: {category}")

    for img in images:
        try:
            img_url = img.get_attribute("src")
            if img_url and img_url.startswith("http"):
                file_path = os.path.join(OUTPUT_DIR, category, f"{count}.jpg")
                urllib.request.urlretrieve(img_url, file_path)
                print(f"[SUCCESS] Скачано: {file_path}")
                count += 1
                if count >= NUM_IMAGES:
                    break
        except Exception as e:
            print(f"[ERROR] Ошибка при загрузке: {e}")

    if count == 0:
        print(f"[WARNING] Не удалось скачать изображения для {category}")
    else:
        print(f"[INFO] Загружено {count} изображений для {category}")

for category, query in CATEGORIES.items():
    download_images(category, query)

driver.quit()
print("[INFO] Завершено!")
