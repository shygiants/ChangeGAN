""" Crawler for collecting photos of clothes from sites of clothing companies """

import time
import ConfigParser

from selenium import webdriver

config = ConfigParser.ConfigParser()
config.read('config/config.conf')


class Crawler:
    def __init__(self):
        self.driver = webdriver.Chrome(config.get('selenium', 'chrome_dir'))
        # Wait for 3 seconds to load resources from the server
        self.driver.implicitly_wait(3)

    def open(self, url):
        self.driver.get(url)

    def find_elements(self, css_selector):
        return self.driver.find_elements_by_css_selector(css_selector)

    def find_element(self, css_selector):
        return self.driver.find_element_by_css_selector(css_selector)

    def execute_script(self, script):
        self.driver.execute_script(script)

    def infinite_scroll(self):
        # Scroll to get all the data because of infinite scrolling
        def get_scroll_height():
            return self.driver.execute_script('return document.body.scrollHeight')

        last_height = get_scroll_height()
        while True:
            self.driver.execute_script('window.scrollTo(0, document.body.scrollHeight);')
            time.sleep(1.5)
            new_height = get_scroll_height()
            if new_height == last_height:
                break
            last_height = new_height

    def go_to_top(self):
        self.driver.execute_script('window.scrollTo(0, 0);')
