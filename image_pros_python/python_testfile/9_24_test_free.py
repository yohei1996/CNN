import numpy as np
import datetime as dt
import os
dt_now = dt.datetime.now()
file_name_date=dt_now.strftime('%Y_%m_%d_')
path = './'
path = os.path.join(path,file_name_date +'model.ckpt')
print(path)