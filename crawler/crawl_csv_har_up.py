import time
import json
from pathlib import Path
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

TARGET_LINK_TEXTS = ["DataSet"]
OUTPUT_FILE = Path(__file__).with_name("har_up_dataset_links.json")

# =================================================================
# 1. CẤU HÌNH TRÌNH DUYỆT GOOGLE CHROME
# =================================================================
options = webdriver.ChromeOptions()
# options.add_argument('--headless') # Bỏ dấu # nếu muốn chạy ẩn ngầm
options.add_argument('--no-sandbox')
options.add_argument('--disable-dev-shm-usage')

driver = webdriver.Chrome(options=options)

def brute_force_iframe_lock(driver):
    """Thuật toán quét sâu 3 tầng iframe để khóa đúng phân vùng Dashboard"""
    print("Đang chạy thuật toán dò tìm 3 tầng Iframe...")
    driver.switch_to.default_content()
    
    iframes_l1 = driver.find_elements(By.TAG_NAME, "iframe")
    for i, frame1 in enumerate(iframes_l1):
        try:
            driver.switch_to.default_content()
            driver.switch_to.frame(frame1)
            if driver.find_elements(By.XPATH, "//*[contains(text(), 'Subject11')]"):
                print(f"[BINGO] Đã khóa mục tiêu tại Iframe Tầng 1 (Index: {i})")
                return True
            
            iframes_l2 = driver.find_elements(By.TAG_NAME, "iframe")
            for j, frame2 in enumerate(iframes_l2):
                try:
                    driver.switch_to.frame(frame2)
                    if driver.find_elements(By.XPATH, "//*[contains(text(), 'Subject11')]"):
                        print(f"[BINGO] Đã khóa mục tiêu tại Iframe Tầng 2 (L1 [{i}] -> L2 [{j}])")
                        return True
                    
                    iframes_l3 = driver.find_elements(By.TAG_NAME, "iframe")
                    for k, frame3 in enumerate(iframes_l3):
                        try:
                            driver.switch_to.frame(frame3)
                            if driver.find_elements(By.XPATH, "//*[contains(text(), 'Subject11')]"):
                                print(f"[BINGO] Đã khóa mục tiêu tại Iframe Tầng 3 (L1 [{i}] -> L2 [{j}] -> L3 [{k}])")
                                return True
                            driver.switch_to.parent_frame()
                        except: continue
                    driver.switch_to.parent_frame()
                except: continue
        except: continue
    return False

try:
    # =================================================================
    # 2. KẾT NỐI VÀ KHÓA IFRAME
    # =================================================================
    print("Đang kết nối tới trang web HAR-UP...")
    driver.get("https://sites.google.com/up.edu.mx/har-up/")
    print("Chờ 12 giây cho hệ thống nạp dữ liệu...")
    time.sleep(12)

    if not brute_force_iframe_lock(driver):
        print("[!] Thất bại: Không thể định vị phân vùng dữ liệu.")
        driver.quit()
        exit()

    # =================================================================
    # 3. TIẾN HÀNH CÀO DATA BẰNG XPATH ĐỊNH DANH (SCOPED XPATH)
    # =================================================================
    all_links = []
    subjects = ["Subject7", "Subject8", "Subject9"]      
    activities = [f"Activity{i}" for i in range(1, 12)]  
    trials = [f"Trial{i}" for i in range(1, 4)]          

    for sub in subjects:
        try:
            print(f"\n====> [BƯỚC 1] Đang chọn: {sub}")
            # Tìm nút Subject ở menu chính ngoài cùng
            sub_btn = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.XPATH, f"//button[contains(., '{sub}')] | //*[text()='{sub}']"))
            )
            driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", sub_btn)
            time.sleep(0.5)
            sub_btn.click()
            print(f"      [✓] Đã click thành công: {sub}")
            time.sleep(1.5) 
        except Exception as e:
            print(f" [!] LỖI không chọn được {sub}: {type(e).__name__}")
            continue 

        for act in activities:
            try:
                print(f"  ---> [BƯỚC 2] Đang chọn: {act}")
                # CHIẾN THUẬT: Chỉ tìm nút Activity NẰM TRONG div của Subject hiện tại
                act_btn = WebDriverWait(driver, 6).until(
                    EC.element_to_be_clickable((By.XPATH, f"//div[@id='{sub}']//button[contains(., '{act}')]"))
                )
                driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", act_btn)
                time.sleep(0.3)
                act_btn.click()
                print(f"        [✓] Đã click thành công: {act}")
                time.sleep(1.0) 
            except Exception as e:
                print(f"   [!] LỖI không chọn được {act}: {type(e).__name__}")
                continue

            for tri in trials:
                try:
                    print(f"    - [BƯỚC 3] Đang chọn: {tri}")
                    # CHIẾN THUẬT: Chỉ tìm nút Trial NẰM TRONG div của Activity hiện tại (Ví dụ: id='Subject11Activity1')
                    tri_btn = WebDriverWait(driver, 5).until(
                        EC.element_to_be_clickable((By.XPATH, f"//div[@id='{sub}{act}']//button[contains(., '{tri}')]"))
                    )
                    driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", tri_btn)
                    time.sleep(0.3)
                    tri_btn.click()
                    print(f"          [✓] Đã click thành công: {tri}")
                    time.sleep(1.5) 
                    
                    # CHIẾN THUẬT: Chỉ tìm các thẻ <a> NẰM TRONG div kết quả hiển thị của Trial đó
                    trial_container = driver.find_element(By.XPATH, f"//div[@id='{sub}{act}{tri}']")
                    a_elements = trial_container.find_elements(By.TAG_NAME, "a")
                    
                    found_any = False
                    for a in a_elements:
                        text = a.text.strip()
                        href = a.get_attribute("href")
                        
                        if text in TARGET_LINK_TEXTS and href:
                            link_data = {
                                "subject": sub,
                                "activity": act,
                                "trial": tri,
                                "file_type": text,
                                "url": href
                            }
                            print(f"            [OK] Bắt được link: {text} -> {href[:50]}...")
                            all_links.append(link_data)
                            found_any = True
                            
                    if not found_any:
                        print("            [?] Tổ hợp này không hiển thị link DataSet.")
                        
                except Exception as e:
                    print(f"     [!] LỖI ở {tri}: {type(e).__name__}")
                    continue

    # =================================================================
    # 4. XUẤT FILE JSON KẾT QUẢ
    # =================================================================
    with OUTPUT_FILE.open("w", encoding="utf-8") as f:
        json.dump(all_links, f, indent=4, ensure_ascii=False)
        
    print("\n" + "="*50)
    print(f"THÀNH CÔNG RỰC RỠ! Đã gom trọn vẹn {len(all_links)} link vào file '{OUTPUT_FILE}'.")
    print("="*50)

finally:
    driver.quit()
