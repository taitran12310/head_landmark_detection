# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 11:27:00 2022

@author: taitran12310
"""

import requests
from zipfile import ZipFile
from io import BytesIO

def download_dataset(save_path, link_download):
    r = requests.get(link_download)
    print("Downloading...")
    z = ZipFile(BytesIO(r.content))
    print("Save zip file")
    output = open(save_path+'/ISBI2015.zip', 'wb')
    output.write(r.content)
    output.close()
    print("Extract zip file")
    z.extractall(save_path + "/")
    print("Completed...")
