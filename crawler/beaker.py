""" Crawl photos from a site of Beaker """

import os

from crawler import Crawler
from utils import img2src, download_images


def main():
    crawler = Crawler()

    def get_photos_from_url(gender, type, url):
        crawler.open(url)

        images = crawler.find_elements('span.front'
                                       '> img')
        srcs = img2src(images)
        print '{} images got'.format(len(images))
        download_images(srcs, os.path.join('beaker', gender, type))

    # Men
    for i in range(27):
        get_photos_from_url('men', 'tshirts-{}'.format(i), 'http://www.ssfshop.com/BEAKER/SFMA42A01/list?dspCtgryNo=SFMA42A01&filterCtgryNo=&secondFilterCtgryNo=&brandShopNo=BDMA09&brndShopId=MCBR&currentPage={}&sortColumn=NEW_GOD_SEQ&etcCtgryNo=&leftBrandNM=BEAKER_MCBR'.format(i+1))
    for i in range(6):
        get_photos_from_url('men', 'shirts-{}'.format(i), 'http://www.ssfshop.com/BEAKER/SFMA42A02/list?dspCtgryNo=SFMA42A02&filterCtgryNo=&secondFilterCtgryNo=&brandShopNo=BDMA09&brndShopId=MCBR&currentPage={}&sortColumn=NEW_GOD_SEQ&etcCtgryNo=&leftBrandNM=BEAKER_MCBR'.format(i+1))
    for i in range(3):
        get_photos_from_url('men', 'knitwear-{}'.format(i), 'http://www.ssfshop.com/BEAKER/SFMA42A03/list?dspCtgryNo=SFMA42A03&filterCtgryNo=&secondFilterCtgryNo=&brandShopNo=BDMA09&brndShopId=MCBR&currentPage={}&sortColumn=NEW_GOD_SEQ&etcCtgryNo=&leftBrandNM=BEAKER_MCBR'.format(i+1))

    # Women
    for i in range(5):
        get_photos_from_url('women', 'shirts-{}'.format(i), 'http://www.ssfshop.com/BEAKER/SFMA41A02/list?dspCtgryNo=SFMA41A02&filterCtgryNo=&secondFilterCtgryNo=&brandShopNo=BDMA09&brndShopId=MCBR&currentPage={}&sortColumn=NEW_GOD_SEQ&etcCtgryNo=&leftBrandNM=BEAKER_MCBR'.format(i+1))
    for i in range(21):
        get_photos_from_url('women', 'tshirts-{}'.format(i), 'http://www.ssfshop.com/BEAKER/SFMA41A01/list?dspCtgryNo=SFMA41A01&filterCtgryNo=&secondFilterCtgryNo=&brandShopNo=BDMA09&brndShopId=MCBR&currentPage={}&sortColumn=NEW_GOD_SEQ&etcCtgryNo=&leftBrandNM=BEAKER_MCBR'.format(i+1))
    for i in range(2):
        get_photos_from_url('women', 'knitwear-{}'.format(i), 'http://www.ssfshop.com/BEAKER/SFMA41A03/list?dspCtgryNo=SFMA41A03&filterCtgryNo=&secondFilterCtgryNo=&brandShopNo=BDMA09&brndShopId=MCBR&currentPage={}&sortColumn=NEW_GOD_SEQ&etcCtgryNo=&leftBrandNM=BEAKER_MCBR'.format(i+1))


if __name__ == '__main__':
    main()
