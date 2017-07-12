""" Crawl photos from a site of American Eagle """

import os

from crawler import Crawler
from utils import img2srcset, download_images


def main():
    crawler = Crawler()

    def get_photos_from_url(gender, type, url):
        crawler.open(url)
        button = crawler.find_element('button.cookie-compliance-ok-btn')
        if button is not None:
            button.click()

        crawler.infinite_scroll()
        if gender == 'men':
            images = crawler.find_elements('img.product-image-front.active')
        else:
            images = crawler.find_elements('img.product-image-front.active')
            # images = crawler.find_elements('img.product-image-reverse')
        srcs = img2srcset(images)
        srcs = filter(lambda src: src.strip() != '', srcs)
        srcs = map(lambda src: 'http:{}'.format(src.split(',')[-1].strip().split(' ')[0]), srcs)
        print '{} images got'.format(len(srcs))
        download_images(srcs, os.path.join('americaneagle', gender, type))

    # Men
    # get_photos_from_url('men', 'tshirts', 'https://www.ae.com/men-t-shirts/web/s-cat/90012?cm=sKR-cUSD&navdetail=mega:mens:c2:p2')
    # get_photos_from_url('men', 'tees', 'https://www.ae.com/men-graphic-tees/web/s-cat/90018?cm=sKR-cUSD')
    # get_photos_from_url('men', 'tanktops', 'https://www.ae.com/men-tank-tops/web/s-cat/7290046?cm=sKR-cUSD')
    # get_photos_from_url('men', 'shirts', 'https://www.ae.com/men-shirts/web/s-cat/40005?cm=sKR-cUSD')
    # get_photos_from_url('men', 'polos', 'https://www.ae.com/men-polos/web/s-cat/3130041?cm=sKR-cUSD')
    # get_photos_from_url('men', 'sweatshirts', 'https://www.ae.com/men-hoodies-sweatshirts/web/s-cat/90020?cm=sKR-cUSD')
    # Women
    get_photos_from_url('women', 'tanktops', 'https://www.ae.com/women-tank-tops/web/s-cat/380157?cm=sKR-cUSD&navdetail=mega:womens:c2:p2')
    get_photos_from_url('women', 'tshirts', 'https://www.ae.com/women-t-shirts/web/s-cat/90030?cm=sKR-cUSD')
    get_photos_from_url('women', 'tees', 'https://www.ae.com/women-graphic-tees/web/s-cat/90042?cm=sKR-cUSD')
    get_photos_from_url('women', 'shirts', 'https://www.ae.com/women-shirts/web/s-cat/90038?cm=sKR-cUSD')
    get_photos_from_url('women', 'sweatshirts', 'https://www.ae.com/women-hoodies-sweatshirts/web/s-cat/90048?cm=sKR-cUSD')



if __name__ == '__main__':
    main()
