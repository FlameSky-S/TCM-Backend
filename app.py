import os
import json
import time
import pickle
import sys
from datetime import datetime
from glob import glob
import numpy as np
import pandas as pd
import logging
import logging.handlers

from flask import Flask
from tqdm import tqdm
from config import *


app = Flask(__name__)

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

# fh = logging.FileHandler(os.path.join('.log', file_name))
fh = logging.handlers.RotatingFileHandler(os.path.join('./log', LOG_FILE_NAME), maxBytes=2e7, backupCount=5)
fh_formatter = logging.Formatter('%(asctime)s - %(name)s - %(message)s')
fh.setFormatter(fh_formatter)
fh.setLevel(logging.DEBUG)
logger.addHandler(fh)

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch_formatter = logging.Formatter('%(name)s - %(message)s')
ch.setFormatter(ch_formatter)
logger.addHandler(ch)


@app.route('/data/getDatasetList', methods=['GET'])
def getDatasetList():
    logger.debug("API called: /data/getDatasetList")
    try:
        df = pd.read_csv(os.path.join(DATA_PATH, 'stat.csv'))
        data_list = []
        # TODO: parse stat.csv & return data_id, data_name, data_num, modalities
        return {"code": SUCCESS_CODE, "msg": "success", "data": data_list}
    except Exception as e:
        # type, value, traceback = sys.exc_info()
        logger.error(e)

@app.route('/data/getDatasetDetails', methods=['GET'])
def getDatasetDetails():
    logger.debug("API called: /data/getDatasetDetails")
    try:
        # TODO: 
        data_list = []
        return {"code": SUCCESS_CODE, "msg": "success", "data": data_list}
    except Exception as e:
        logger.error(e)

@app.route('/features/face/supported', methods=['GET'])
def face_supported():
    logger.debug("API called: /features/face/supported")
    try:
        pass
    # TODO: 
    except Exception as e:
        logger.error(e)



app.run(port=5000)