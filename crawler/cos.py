""" Crawl photos from a site of COS """

import os

from crawler import Crawler
from utils import download_images


def main():
    crawler = Crawler()

    def get_photos_from_url(gender, type, url):
        crawler.open(url)
        crawler.infinite_scroll(sleep_time=3.)
        images = crawler.find_elements('figure > noscript')

        def parse_src(elem):
            html = elem.get_attribute('innerHTML')
            for token in html.split(' '):
                if 'src' in token:
                    src = token.split('"')[-2]
                    return 'http://www.cosstores.com{}'.format(src)
        srcs = map(parse_src, images)
        print '{} images got'.format(len(images))
        download_images(srcs, os.path.join('cos', gender, type))

    # Men
    # get_photos_from_url('men', 'shirts', 'http://www.cosstores.com/us/Men/Shirts')
    # get_photos_from_url('men', 'tees', 'http://www.cosstores.com/us/Men/T-shirts_Polo_shirts')
    # get_photos_from_url('men', 'sweatshirts', 'http://www.cosstores.com/us/Men/Sweatshirts_Tops')
    # get_photos_from_url('men', 'knitwear', 'http://www.cosstores.com/us/Men/Knitwear')

    # Women
    get_photos_from_url('women', 'tops', 'http://www.cosstores.com/us/Women/Tops')
    get_photos_from_url('women', 'tees', 'http://www.cosstores.com/us/Women/T-shirts_Sweatshirts')
    get_photos_from_url('women', 'shirts', 'http://www.cosstores.com/us/Women/Shirts')
    get_photos_from_url('women', 'knitwear', 'http://www.cosstores.com/us/Women/Knitwear')


if __name__ == '__main__':
    main()
