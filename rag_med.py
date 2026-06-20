import os 
os.environ["CUDA_VISIBLE_DEVICEs"] = "0"

import torch as ts 
ts.cuda.get_device_name(0)

from langchain_community.document_loaders import DirectoryLoader , PyPDFLoader

