""" Crawl photos from a site of Gap """

import os

from crawler import Crawler
from utils import img2src, download_images


def main():
    crawler = Crawler()

    def get_photos_from_url(gender, type, url):
        crawler.open(url)
        crawler.infinite_scroll(slowly=True)
        images = crawler.find_elements('img.product-card--img')
        srcs = img2src(images)
        print '{} images got'.format(len(images))
        download_images(srcs, os.path.join('gap', gender, type))

    # Men
    # get_photos_from_url('men', 'tee', 'http://www.gap.com/browse/category.do?cid=5225&sop=true&departmentRedirect=true#pageId=0&department=75')
    # get_photos_from_url('men', 'tees', 'http://www.gap.com/browse/category.do?cid=1059955&sop=true&departmentRedirect=true#pageId=0&department=75')
    # get_photos_from_url('men', 'shirts', 'http://www.gap.com/browse/category.do?cid=15043&sop=true&departmentRedirect=true#pageId=0&department=75')
    # get_photos_from_url('men', 'sweaters', 'http://www.gap.com/browse/category.do?cid=5180&sop=true&departmentRedirect=true#pageId=0&department=75')
    # get_photos_from_url('men', 'sweatshirts', 'http://www.gap.com/browse/category.do?cid=1066503&sop=true&departmentRedirect=true#pageId=0&department=75')

    # Women
    # get_photos_from_url('women', 'tees', 'http://www.gap.com/browse/category.do?cid=17076&sop=true&departmentRedirect=true#pageId=0&department=136')
    get_photos_from_url('women', 'shirts', 'http://www.gap.com/browse/category.do?cid=34608&sop=true&departmentRedirect=true#pageId=0&department=136')
    get_photos_from_url('women', 'sweaters', 'http://www.gap.com/browse/category.do?cid=5745&sop=true&departmentRedirect=true#pageId=0&department=136')
    get_photos_from_url('women', 'sweatshirts', 'http://www.gap.com/browse/category.do?cid=1041168&sop=true&departmentRedirect=true#pageId=0&department=136')


if __name__ == '__main__':
    main()
