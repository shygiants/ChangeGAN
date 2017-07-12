""" Crawl photos from a site of Giorgio Armani """

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
        download_images(srcs, os.path.join('giorgioarmani', gender, type))

    # Men
    # get_photos_from_url('men', 'shirts', 'http://www.armani.com/us/giorgioarmani/men/onlinestore/cmcga')
    get_photos_from_url('men', 'knitwear', 'http://www.armani.com/us/giorgioarmani/men/onlinestore/knitwear')
    get_photos_from_url('men', 'tshirts', 'http://www.armani.com/us/giorgioarmani/men/onlinestore/tshrtga')

    # Women
    get_photos_from_url('women', 'tops', 'http://www.armani.com/us/giorgioarmani/women/onlinestore/t-shirts-and-shirts')
    get_photos_from_url('women', 'knitwear', 'http://www.armani.com/us/giorgioarmani/women/onlinestore/knitwear')


if __name__ == '__main__':
    main()
