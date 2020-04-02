from flickrapi import FlickrAPI
import os
from urllib.request import urlretrieve
import time

flickr_public = "
flickr_secret = ""
wait_time = 1

words = ["car", "train", "ship", "airplane", "bicycle"]

for word in words:
    os.mkdir("./Images/{}".format(word))

for word in words:
    path = "./Images/{}".format(word)
    flickr = FlickrAPI(flickr_public, flickr_secret, format="parsed-json")
    result = flickr.photos.search(
        text = word,
        per_page = 300,
        media = "photos",
        sort = "relevance",
        safe_search = 1,
        extras = "url_q, lisence"
    )

    photos = result["photos"]

    for i, photo in enumerate(photos["photo"]):
        url_q = photo["url_q"]
        file_path = path + "/" + photo["id"] + ".jpg"

        if os.path.exists(file_path):
            continue

        urlretrieve(url_q, file_path)
        time.sleep(wait_time)
