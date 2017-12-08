"""

Class for interfacing with the flickr API and retrieving image download links.
You need the keys from Flickr in order to connect to their API.
This class requires the flickrapi python package (pip install flickrapi)

John Kwong
12/7/2017

"""
from flickrapi import FlickrAPI
import numpy as np

class FlickrFetchImages(object):

    def __init__(self, public_key, secret_key):
        self.flickr = FlickrAPI(public_key, secret_key, format='parsed-json')

    def GetImageUrls(self,search_key, per_page = 100, number_pages = 1):
        extras = 'url_sq','url_t','url_s','url_q','url_m','url_n','url_z','url_c','url_l','url_o'

        self.url_dict_list = {e:[] for e in extras}

        for page_number in xrange(number_pages):

            results = self.flickr.photos.search(text=search_key, \
                                                per_page=per_page, page = page_number, sort = 'relevance', extras=','.join(extras))
            photos = results['photos']['photo']
            for url_type in extras:
                for p in photos:
                    try:
                        self.url_dict_list[url_type].append(p[url_type])
                    except:
                        print('skip')

    def GetRandomImage(self, search_key, image_type = None):
        MAX_PAGE_NUMBER = 5
        PER_PAGE = 50

        # extras = 'url_sq','url_t','url_s','url_q','url_m','url_n','url_z','url_c','url_l','url_o'
        # we only get urls for these images
        extras = 'url_o','url_n'

        self.url_dict_list = {e:[] for e in extras}
        page_number = np.random.randint(0, MAX_PAGE_NUMBER)

        results = self.flickr.photos.search(text=search_key, per_page=PER_PAGE, page = page_number, \
                                            sort = 'relevance',\
                                            extras=','.join(extras))
        while 1:
            try:
                image_index = np.random.randint(0, len(results['photos']['photo']))
                # if image of particular type is not present, then try again
                if image_type is not None:
                    if len(results['photos']['photo'][image_index][image_type]) == 0:
                        continue

                a_photo = results['photos']['photo'][image_index]
                break
            except:
                continue
        return a_photo
