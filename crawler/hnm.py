""" Crawl photos from a site of H&M """

import os

from crawler import Crawler
from utils import img2src, download_images


def main():
    crawler = Crawler()

    def get_photos_from_url(gender, type, url):
        # Go to site of H&M
        crawler.open(url)

        def get_product_images(dir):
            crawler.infinite_scroll()
            images = crawler.find_elements('article.product-item > a > img')
            print '{} images got'.format(len(images))
            srcs = img2src(images)
            download_images(srcs, os.path.join('hnm', gender, type, dir))

        # Get images of the models
        get_product_images('models')

        # Go to top
        crawler.go_to_top()
        crawler.find_element('li > a#color_filter_stillLife').click()

        # Get images of the clothes
        get_product_images('clothes')

    # Men
    get_photos_from_url('men', 'shirts-and-tanks',
                        'http://www2.hm.com/ko_kr/men/shop-by-product/shirts-and-tanks.html')
    get_photos_from_url('men', 'shirts',
                        'http://www2.hm.com/ko_kr/men/shop-by-product/shirts.html')
    get_photos_from_url('men', 'hoodies-and-sweatshirts',
                        'http://www2.hm.com/ko_kr/men/shop-by-product/hoodies-and-sweatshirts.html')

    # Women
    get_photos_from_url('women', 'shirts-and-blouses',
                        'http://www2.hm.com/ko_kr/ladies/shop-by-product/shirts-and-blouses.html')
    get_photos_from_url('women', 'tops',
                        'http://www2.hm.com/ko_kr/ladies/shop-by-product/tops.html')
    get_photos_from_url('women', 'hoodies-and-sweatshirts',
                        'http://www2.hm.com/ko_kr/ladies/shop-by-product/cardigans-and-jumpers/hoodies-and-sweatshirts.html')


if __name__ == '__main__':
    main()
