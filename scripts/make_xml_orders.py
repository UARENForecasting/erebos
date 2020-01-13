import os
from pathlib import Path
from selenium.webdriver import Firefox, FirefoxOptions, Chrome, ChromeOptions


def innerhtml(driver):
    return driver.find_element_by_xpath("/html/body").get_attribute("innerHTML")


opts = ChromeOptions()
opts.add_argument("--disable-extensions")
opts.add_argument("--disable-dev-shm-model")
opts.add_argument("--no-sandbox")
opts.add_argument("--remote-debugging-port=9222")
opts.headless = True
# opts.profile = '/home/alorenzo/safeprofile/'
# driver = Firefox(options=opts)
driver = Chrome(options=opts)
driver.get("https://www.class.noaa.gov")
class_url = Path(driver.current_url)
# remove https:/ and end paths
class_url = class_url.parents[2].relative_to(class_url.parents[3])
login_url = class_url / "saa/products/classlogin?resource=%2Fsaa%2Fproducts%2Fupload"


driver.get("https://" + str(login_url))
user_field = driver.find_element_by_name("j_username")
pass_field = driver.find_element_by_name("j_password")
submit = driver.find_element_by_xpath("/html/body/div[1]/div[5]/div/form/input[3]")

# login
user_field.click()
user_field.send_keys(os.getenv("CLASSUSER"))
pass_field.click()
pass_field.send_keys(os.getenv("CLASSPASS"))
submit.click()
assert "upload" in driver.current_url

xml_base = Path("/storage/projects/goes_alg/goes_data/west/xml")


def add_to_cart(driver, xml_file):
    driver.get("https://" + str(class_url / "saa/products/upload"))
    browse = driver.find_element_by_name("uploaded_file")
    browse.send_keys(str(xml_file))
    driver.find_element_by_xpath(
        "/html/body/div[1]/div[5]/content/center/i/form/p/input"
    ).click()
    assert "search" in driver.current_url

    search_btn = driver.find_element_by_id("searchbutton")
    search_btn.click()

    assert "psearch" in driver.current_url

    driver.find_element_by_xpath(
        "/html/body/div[1]/div[5]/page/div[2]/form[2]/table[1]/tbody/tr/td[2]/input[3]"
    ).click()
    driver.get("https://" + str(class_url / "saa/products/shopping_cart"))
    num_sets = driver.find_element_by_xpath(
        "/html/body/div[1]/div[5]/form[2]/table/tbody/tr[2]/td[2]"
    ).text
    xml_file.rename(xml_file.parent / "maybe_processing" / xml_file.name)
    return int(num_sets)


num_sets = 0
for xml_file in xml_base.glob("*.xml"):
    new_sets = add_to_cart(driver, xml_file)
    print(new_sets)
    assert new_sets > num_sets
    num_sets = new_sets
    if num_sets >= 100:
        break

driver.find_element_by_xpath("/html/body/div[1]/div[5]/form[2]/div[2]/input[1]").click()
for afile in (xml_base / "maybe_processing").glob("*.xml"):
    afile.rename(xml_base / "processing" / afile.name)
driver.close()
