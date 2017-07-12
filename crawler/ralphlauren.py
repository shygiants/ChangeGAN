""" Crawl photos from a site of Ralph Lauren """

import os

from crawler import Crawler
from utils import img2src, download_images


def main():
    crawler = Crawler()

    def get_photos_from_url(gender, type, url):
        crawler.open(url)

        images = crawler.find_elements('a.photo '
                                       '> img')
        srcs = img2src(images)
        print '{} images got'.format(len(images))
        download_images(srcs, os.path.join('larphlauren', gender, type))

    # Men
    # get_photos_from_url('men', 'shirts', 'http://www.ralphlauren.com/family/index.jsp?categoryId=2498319&cp=1760781&ab=ln_men_cs_casualshirts')
    # get_photos_from_url('men', 'tshirts','http://www.ralphlauren.com/family/index.jsp?categoryId=1760812&cp=1760781&ab=ln_men_cs_t-shirts&sweatshirts')
    # get_photos_from_url('men', 'sweaters','http://www.ralphlauren.com/family/index.jsp?categoryId=1760813&cp=1760781&ab=ln_men_cs_sweaters')
    # Women
    # get_photos_from_url('women', 'shirts', 'http://www.ralphlauren.com/family/index.jsp?categoryId=57940726&cp=1760782&ab=ln_women_cs_shirts&tops')
    get_photos_from_url('women', 'polo', 'http://www.ralphlauren.com/family/index.jsp?categoryId=114979276&cp=1760782&ab=ln_women_cs_poloshirts')
    get_photos_from_url('women', 'sweaters', 'http://www.ralphlauren.com/family/index.jsp?categoryId=1760895&cp=1760782&ab=ln_women_cs_sweaters')


if __name__ == '__main__':
    main()
