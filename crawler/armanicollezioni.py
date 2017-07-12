""" Crawl photos from a site of Armani Collezioni """

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
        download_images(srcs, os.path.join('armanicollezioni', gender, type))

    # Men
    # get_photos_from_url('men', 'shirts', 'http://www.armani.com/us/armanicollezioni/men/onlinestore/shirts')
    # get_photos_from_url('men', 'knitwear', 'http://www.armani.com/us/armanicollezioni/men/onlinestore/knitwear')
    get_photos_from_url('men', 'tshirts', 'http://www.armani.com/us/armanicollezioni/men/onlinestore/tshrtssm')
    get_photos_from_url('men', 'sweatshirts', 'http://www.armani.com/us/armanicollezioni/men/onlinestore/flpss')

    # Women
    get_photos_from_url('women', 'tops', 'http://www.armani.com/us/armanicollezioni/women/onlinestore/tops')
    get_photos_from_url('women', 'tshirts', 'http://www.armani.com/us/armanicollezioni/women/onlinestore/tshrtssd')
    get_photos_from_url('women', 'knitwear', 'http://www.armani.com/us/armanicollezioni/women/onlinestore/knitwear')
    get_photos_from_url('women', 'sweatshirts', 'http://www.armani.com/us/armanicollezioni/women/onlinestore/flpss')


if __name__ == '__main__':
    main()
