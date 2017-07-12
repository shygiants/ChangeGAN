""" Crawl photos from a site of Emprio Armani """

import os

from crawler import Crawler
from utils import img2src, download_images


def main():
    crawler = Crawler()

    def get_photos_from_url(gender, type, url):
        crawler.open(url)
        crawler.infinite_scroll(slowly=True)
        images = crawler.find_elements('img.photo.default')
        srcs = img2src(images)
        print '{} images got'.format(len(images))
        download_images(srcs, os.path.join('emporioarmani', gender, type))

    # Men
    get_photos_from_url('men', 'shirts', 'http://www.armani.com/us/emporioarmani/men/onlinestore/shirts')
    get_photos_from_url('men', 'knitwear', 'http://www.armani.com/us/emporioarmani/men/onlinestore/knitwear')
    get_photos_from_url('men', 'tshirts', 'http://www.armani.com/us/emporioarmani/men/onlinestore/t-shirts-and-sweatshirts')

    # Women
    get_photos_from_url('women', 'tops', 'http://www.armani.com/us/emporioarmani/women/onlinestore/tops')
    get_photos_from_url('women', 'tshirts', 'http://www.armani.com/us/emporioarmani/women/onlinestore/t-shirts-and-sweatshirts')
    get_photos_from_url('women', 'knitwear', 'http://www.armani.com/us/emporioarmani/women/onlinestore/knitwear')


if __name__ == '__main__':
    main()
