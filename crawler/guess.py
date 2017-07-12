""" Crawl photos from a site of Guess """

import os

from crawler import Crawler
from utils import img2src, download_images


def main():
    crawler = Crawler()

    def get_photos_from_url(gender, type, url):
        crawler.open(url)
        crawler.infinite_scroll(slowly=True)
        images = crawler.find_elements('div.item-imgwrap'
                                       '> img:not(.hover)')
        srcs = img2src(images)
        print '{} images got'.format(len(images))
        download_images(srcs, os.path.join('guess', gender, type))

    # Men
    # get_photos_from_url('men', 'top', 'https://www.guesskorea.com/category/men/top')

    # Women
    get_photos_from_url('women', 'top', 'https://www.guesskorea.com/category/women/top')


if __name__ == '__main__':
    main()
