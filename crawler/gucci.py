""" Crawl photos from a site of Gucci """

import os

from crawler import Crawler
from utils import img2src, download_images


def main():
    crawler = Crawler()

    def get_photos_from_url(gender, type, url):
        crawler.open(url)

        images = crawler.find_elements('a.product-tiles-grid-item-link '
                                       '> div.product-tiles-grid-item-image-wrapper '
                                       '> div.product-tiles-grid-item-image '
                                       '> img')
        srcs = img2src(images)
        print '{} images got'.format(len(images))
        download_images(srcs, os.path.join('gucci', gender, type))

    # Men
    # get_photos_from_url('men', 'shirts', 'https://www.gucci.com/kr/ko/ca/men/mens-ready-to-wear/mens-shirts-c-men-readytowear-shirts')

    # Women
    # get_photos_from_url('women', 'shirts', 'https://www.gucci.com/kr/ko/ca/women/womens-ready-to-wear/womens-tops-shirts-c-women-readytowear-tops-and-shirts')
    get_photos_from_url('women', 'sweatshirts', 'https://www.gucci.com/kr/ko/ca/women/womens-ready-to-wear/womens-sweatshirts-t-shirts-c-women-ready-to-wear-new-sweatshirts')


if __name__ == '__main__':
    main()
