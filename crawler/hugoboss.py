""" Crawl photos from a site of Hugo Boss """

import os

from crawler import Crawler
from utils import img2src, download_images


def main():
    crawler = Crawler()

    def get_photos_from_url(gender, type, url):
        crawler.open(url)

        crawler.infinite_scroll(sleep_time=3.5)
        images = crawler.find_elements('img.product-tile__image')
        srcs = img2src(images)
        print '{} images got'.format(len(images))
        download_images(srcs, os.path.join('hugoboss', gender, type))

    # Men
    # get_photos_from_url('men', 'shirts', 'http://www.hugoboss.com/asia/men-shirts/')
    # get_photos_from_url('men', 'polo', 'http://www.hugoboss.com/asia/men-polo-shirts/')
    # get_photos_from_url('men', 'tshirts', 'http://www.hugoboss.com/asia/men-t-shirts/')
    # get_photos_from_url('men', 'knitwear', 'http://www.hugoboss.com/asia/men-knitwear')
    # get_photos_from_url('men', 'sweatshirts', 'http://www.hugoboss.com/asia/men-sweatshirts/')

    # Women
    # get_photos_from_url('women', 'knitwear', 'http://www.hugoboss.com/asia/women-knitwear/')
    # get_photos_from_url('women', 'blouses', 'http://www.hugoboss.com/asia/women-blouses/')
    # get_photos_from_url('women', 'tops', 'http://www.hugoboss.com/asia/women-tops/')
    get_photos_from_url('women', 'tshirts', 'http://www.hugoboss.com/asia/women-t-shirts/')


if __name__ == '__main__':
    main()
