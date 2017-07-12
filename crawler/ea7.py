""" Crawl photos from a site of EA7 """

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
        download_images(srcs, os.path.join('ea7', gender, type))

    # Men
    get_photos_from_url('men', 'sweatshirts', 'http://www.armani.com/us/ea7/men/onlinestore/flpea7')
    get_photos_from_url('men', 'tshirts', 'http://www.armani.com/us/ea7/men/onlinestore/t-shirts')

    # Women
    get_photos_from_url('women', 'tshirts', 'http://www.armani.com/us/ea7/women/onlinestore/t-shirts')


if __name__ == '__main__':
    main()
