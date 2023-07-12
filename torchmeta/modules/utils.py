import re   ##检查字符串是否与某种模式匹配
from collections import OrderedDict   ###自动排序

def get_subdict(dictionary, key=None):
    if dictionary is None:
        print("dictionary is None")
        return None
    if (key is None) or (key == ''):
        return dictionary
    key_re = re.compile(r'^{0}\.(.+)'.format(re.escape(key)))   ## re.escape()对字符串中所有可能被解释为正则运算符的字符进行转义。
    return OrderedDict((key_re.sub(r'\1', k), value) for (k, value)   ##
        in dictionary.items() if key_re.match(k) is not None)
