""" Crawl photos from a site of LEBEIGE """

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
        download_images(srcs, os.path.join('lebeige', gender, type))

    # Women
    get_photos_from_url('women', 'shirts', 'http://www.ssfshop.com/LEBEIGE/ssfshop/list?dspCtgryNo=SFMA41A02&brandShopNo=BDMA07A06&brndShopId=ECBVF&etcCtgryNo=&ctgrySectCd=&keyword=&leftBrandNM=LEBEIGE_ECBVF')
    get_photos_from_url('women', 'tshirts', 'http://www.ssfshop.com/LEBEIGE/T-shirts/list?dspCtgryNo=SFMA41A01&brandShopNo=BDMA07A06&brndShopId=ECBVF&etcCtgryNo=&ctgrySectCd=&keyword=&leftBrandNM=LEBEIGE_ECBVF')
    get_photos_from_url('women', 'knitwear', 'http://www.ssfshop.com/LEBEIGE/Knitwear/list?dspCtgryNo=SFMA41A03&brandShopNo=BDMA07A06&brndShopId=ECBVF&etcCtgryNo=&ctgrySectCd=&keyword=&leftBrandNM=LEBEIGE_ECBVF')


if __name__ == '__main__':
    main()
