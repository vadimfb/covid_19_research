import requests
import os
import zipfile

from args import *
from utils import *


def download_url(url, save_path, chunk_size=128):
    r = requests.get(url, stream=True)
    with open(save_path, 'wb') as fd:
        for chunk in r.iter_content(chunk_size=chunk_size):
            fd.write(chunk)


def download_new_data(input_path, output_path):
    zip_file = input_path + 'COVID-19-master.zip'
    unzip_dir = output_path
    download_url('https://github.com/CSSEGISandData/COVID-19/archive/master.zip', zip_file)
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(unzip_dir)
    try:
        os.remove(zip_file)
    except:
        print('Already removed')

        
def check_latest_date():
    path1 = INPUT_PATH + 'COVID-19-master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv'
    s = pd.read_csv(path1)
    print('Latest date available in {}: {}'.format(os.path.basename(path1), s.columns.values[-1]))