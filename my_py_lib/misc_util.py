
import os,time
import numpy as np
from pathlib import Path
class ch_cwd_to_this_file:
    def __init__(self, _code_file_path):  # _code_file_path typically receives __file__
        self._code_file_path = _code_file_path
    def __enter__(self):
        self._old_dir = os.getcwd()
        cwd=os.path.dirname(os.path.abspath(self._code_file_path))
        os.chdir(cwd)
    def __exit__(self, exc_type, exc_val, exc_tb):
        os.chdir(self._old_dir)
# def img_2_img_full_path(img,format='jpg',original_name_or_path=''):
#     """
#     thread safe
#     """
#     assert isinstance(img,np.ndarray)
#     assert img.shape[2]==3 or img.shape[2]==4
#     original_img_name_without_dir=os.path.basename(original_name_or_path)
#     full_path = os.path.join(root_config.path_root, f'./tmp_images/[{root_config.DATASET}][{tmp_cate_or_obj}][{sequence_name}]{img_name_without_suffix}.jpg')
#     if not os.path.exists(os.path.dirname(full_path)):
#         os.makedirs(os.path.dirname(full_path))
#     print("get_data path:", full_path)
#     img.save(full_path)
#     return img_full_path

import datetime
import pytz
def beijing_datetime()->datetime.datetime:
    """
    Example: print(f'Current Beijing time = {beijing_time:%Y.%m.%d %H:%M:%S}')
    """
    # get the local timezone
    local_tz = datetime.datetime.now(datetime.timezone.utc).astimezone().tzinfo
    # get Beijing timezone
    beijing_tz = pytz.timezone('Asia/Shanghai')
    # get the current time
    now = datetime.datetime.now()
    # convert current time to local timezone
    local_time = now.astimezone(local_tz)
    # convert local time to Beijing timezone
    beijing_time:datetime.datetime = local_time.astimezone(beijing_tz)
    return beijing_time
def beijing_str_A( os_is_windows=False)->str:
    """
    print(  beijing_str_A()   )
    """
    ret= f"{beijing_datetime():%m.%d-%H:%M:%S}"
    if os_is_windows:
        ret=ret.replace(':','：')
    return ret





# convert numpy or tensor to json/dict
import json
import numpy
import PIL
import torch
from torch import Tensor


def to_list_to_primitive(obj):
    if isinstance(obj, numpy.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.cpu().data.numpy().tolist()
    if isinstance(obj, list):
        return [to_list_to_primitive(i) for i in obj]
    # if isinstance(obj, DataFrame):
    #     return obj.values.tolist()
    elif (isinstance(obj, numpy.int32) or
          isinstance(obj, numpy.int64) or
          isinstance(obj, numpy.float32) or
          isinstance(obj, numpy.float64)):
        return obj.item()
    elif (isinstance(obj, int) or
          isinstance(obj, float)
          ):
        return obj
    else:
        raise TypeError("got {}".format(type(obj)))
def to_ndarray(x):
    if isinstance(x, numpy.ndarray):
        return x
    if isinstance(x, torch.Tensor):
        return x.cpu().data.numpy()
    if isinstance(x, list):
        return numpy.array(x)
    if isinstance(x, PIL.Image.Image):
        return numpy.array(x)
    # if isinstance(x, int) or isinstance(x, float):
    #     return numpy.array([x])
    raise TypeError("got {}".format(type(x)))

def to_tensor(x):
    if isinstance(x, numpy.ndarray):
        return torch.from_numpy(x)
    if isinstance(x, torch.Tensor):
        return x
    if isinstance(x, PIL.Image.Image):
        return torch.from_numpy(numpy.array(x))
    if isinstance(x, list):
        return torch.tensor(x)
    # if isinstance(x, int) or isinstance(x, float):
    #     return torch.tensor([x])
    raise TypeError("got {}".format(type(x)))
def to_pil(x):
    import torch
    if isinstance(x, PIL.Image.Image):
        return x
    if isinstance(x, numpy.ndarray):
        return PIL.Image.fromarray(x)
    if isinstance(x, torch.Tensor):
        return PIL.Image.fromarray(x.cpu().data.numpy())
    raise TypeError("got {}".format(type(x)))


class myJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.ndarray):
            return obj.tolist()
        if isinstance(obj, Tensor):
            return obj.cpu().data.numpy().tolist()
        elif (isinstance(obj, numpy.int32) or
              isinstance(obj, numpy.int64) or
              isinstance(obj, numpy.float32) or
              isinstance(obj, numpy.float64)):
            return obj.item()
        elif isinstance(obj,Path):
            return str(obj)
        return json.JSONEncoder.default(self, obj)

if(__name__=="__main__"):
    import  torch
    dic = {'x': torch.randn(2, 3), 'rec': numpy.array([[11, 22, 33], [44, 55, 66], [77, 88, 99]])}
    s_dic=json.dumps(dic , cls=myJSONEncoder,
                     sort_keys=True, indent=2,
                     separators=(',', ': '), ensure_ascii=False)
    with open('test.json', 'w', encoding='utf8') as f:
        json.dump(dic,f,
                  # sort_keys=True,
                  sort_keys=False,
                  indent=2, separators=(',', ': '), ensure_ascii=False)
        
        

def truncate_str(string:str,MAX_LEN:int,suffix_if_truncate="......")->str:
    assert isinstance(string,str)
    if len(string)>  MAX_LEN:
        string=string[:MAX_LEN]+suffix_if_truncate
    return string
def map_string_to_int(string,MIN,MAX):
    """
    Map strings evenly into [MIN, MAX]
    """
    assert isinstance(MIN,int)
    assert isinstance(MAX,int)
    assert MAX-MIN>=2
    # compute ASCII sum
    sum = 0
    for char in string:
        sum += ord(char)
    # print("sum", sum)
    ret=2**sum
    ret += sum # avoid producing only powers of two
    ret=ret%(MAX-MIN)
    ret+=MIN
    return ret
if 0:
    import pprint
    def print_optimizer(optimizer):
        state_dict=optimizer.state_dict()
        param_groups=state_dict['param_groups']
        # for i,param_group in enumerate(param_groups):
        pprint.pprint(param_groups)


def dic_key_str_2_int(dic: dict) -> dict:
    ret = {}
    for k, v in dic.items():
        if isinstance(k, str) and k.isdigit():
            k = int(k)
        ret[k] = v
    return ret
def dic_key_str_2_int__nested(dic: dict) -> dict:
    ret = {}
    for k, v in dic.items():
        if isinstance(k, str) and k.isdigit():
            k = int(k)
        if isinstance(v, dict):
            v = dic_key_str_2_int__nested(v)
        ret[k] = v
    return ret
def dic_list_2_tuple_nested(dic: dict) -> dict:#if k,v is list, to tuple
    ret = {}
    for k, v in dic.items():
        if isinstance(k, list):
            k = tuple(k)
        if isinstance(v, list):
            v = tuple(v)
        if isinstance(v, dict):
            v = dic_list_2_tuple_nested(v)
        ret[k] = v
    return ret


import re

def inverse_fstring(string:str,fmt:str,):
    """
    Inverse of string format in python
    from https://stackoverflow.com/questions/48536295/inverse-of-string-format-in-python
    """
    reg_keys = '{([^{}:]+)[^{}]*}'
    reg_fmts = '{[^{}:]+[^{}]*}'
    pat_keys = re.compile(reg_keys)
    pat_fmts = re.compile(reg_fmts)

    keys = pat_keys.findall(fmt)
    lmts = pat_fmts.split(fmt)
    temp = string
    values = []
    for lmt in lmts:
        if not len(lmt)==0:
            value,temp = temp.split(lmt,1)
            if len(value)>0:
                values.append(value)
    if len(temp)>0:
        values.append(temp)
    return dict(zip(keys,values))
def sort_strings_asc_A(l:list,fmt:str)->list:
    """
    fmt: eg. 'home/frame{d}.png'
    """
    ret=sorted(l, key=  lambda s:int( inverse_fstring(s, fmt )['d'])   )
    return ret
from natsort import natsorted
def ls_natsort(folder,re_="*"):
    folder = Path(folder)
    files = list(folder.glob(re_))
    return natsorted(files )
    return natsorted(files, key=lambda x: x.name)



if __name__=='__main__':
    print(  beijing_str_A()  )
    if 1:
        fmt = '{k1:}+{k2:}={k:3}'
        res = '1+1=2'
        print (inverse_fstring(res,fmt))

        fmt = '{name:} {age:} {gender}'
        res = 'Alice 10 F'
        print (inverse_fstring(res,fmt))

        fmt = 'Hi, {k1:}, this is {k2:}'
        res = 'Hi, Alice, this is Bob'
        print (inverse_fstring(res,fmt))
