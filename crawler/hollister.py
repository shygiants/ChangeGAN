""" Crawl photos from a site of Hollister """

import os

from crawler import Crawler
from utils import img2src, download_images


def main():
    crawler = Crawler()

    def get_photos_from_url(gender, type, url):
        crawler.open(url)
        crawler.infinite_scroll()
        images = crawler.find_elements('img.grid-product__image')
        srcs = img2src(images)
        print '{} images got'.format(len(images))
        if gender == 'women':
            srcs = map(lambda src: src.replace('model', 'prod'), srcs)
        download_images(srcs, os.path.join('hollister', gender, type))

    # Men
    # get_photos_from_url('men', 'top', 'https://www.hollisterco.com/shop/wd/guys-tops/?search-field=&sort=newest&start=0&rows=90&filtered=true')
    # get_photos_from_url('men', 'top-1', 'https://www.hollisterco.com/shop/wd/guys-tops/?search-field=&sort=newest&start=90&rows=90&filtered=true')
    # get_photos_from_url('men', 'top-2', 'https://www.hollisterco.com/shop/wd/guys-tops/?search-field=&sort=newest&start=180&rows=90&filtered=true')

    # Women
    # get_photos_from_url('women', 'top', 'https://www.hollisterco.com/shop/wd/girls-tops/?search-field=&sort=newest&start=0&rows=90&filtered=true')
    get_photos_from_url('women', 'top-1', 'https://www.hollisterco.com/shop/wd/girls-tops/?search-field=&sort=newest&start=90&rows=90&filtered=true')


if __name__ == '__main__':
    main()
