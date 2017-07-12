""" Crawl photos from a site of KUHO """

import os

from crawler import Crawler
from utils import img2src, download_images


def main():
    crawler = Crawler()

    def get_photos_from_url(gender, type, url):
        crawler.open(url)

        images = crawler.find_elements('span.back'
                                       '> img')
        srcs = img2src(images)
        print '{} images got'.format(len(images))
        download_images(srcs, os.path.join('kuho', gender, type))

    # Women
    for i in range(2):
        get_photos_from_url('women', 'shirts-{}'.format(i), 'http://www.ssfshop.com/KUHO/SFMA41A02/list?dspCtgryNo=SFMA41A02&filterCtgryNo=&secondFilterCtgryNo=&brandShopNo=BDMA07A02&brndShopId=WMBKF&currentPage={}&sortColumn=NEW_GOD_SEQ&etcCtgryNo=&leftBrandNM=KUHO_WMBKF'.format(i+1))
    for i in range(2):
        get_photos_from_url('women', 'tshirts-{}'.format(i), 'http://www.ssfshop.com/KUHO/SFMA41A01/list?dspCtgryNo=SFMA41A01&filterCtgryNo=&secondFilterCtgryNo=&brandShopNo=BDMA07A02&brndShopId=WMBKF&currentPage={}&sortColumn=NEW_GOD_SEQ&etcCtgryNo=&leftBrandNM=KUHO_WMBKF'.format(i+1))
    for i in range(2):
        get_photos_from_url('women', 'knitwear-{}'.format(i), 'http://www.ssfshop.com/KUHO/SFMA41A03/list?dspCtgryNo=SFMA41A03&filterCtgryNo=&secondFilterCtgryNo=&brandShopNo=BDMA07A02&brndShopId=WMBKF&currentPage={}&sortColumn=NEW_GOD_SEQ&etcCtgryNo=&leftBrandNM=KUHO_WMBKF'.format(i+1))


if __name__ == '__main__':
    main()
