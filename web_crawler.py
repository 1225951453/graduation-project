#????
import urllib.request
import re
import requests

def getDatas(keyword, pages):
    params = []
    for i in range(30, 30 * pages + 30, 30):
        params.append({
            'tn': 'resultjson_com',
            'ipn': 'rj',
            'ct': 201326592,
            'is': '',
            'fp': 'result',
            'queryWord': keyword,
            'cl': 2,
            'lm': -1,
            'ie': 'utf-8',
            'oe': 'utf-8',
            'adpicid': '',
            'st': -1,
            'z': '',
            'ic': 0,
            'word': keyword,
            's': '',
            'se': '',
            'tab': '',
            'width': '',
            'height': '',
            'face': 0,
            'istype': 2,
            'qc': '',
            'nc': 1,
            'fr': '',
            'pn': i, # i
            'rn': 30,
            'gsm': '1e',
            '1526377465547': ''
        })
    url = 'https://image.baidu.com/search/index'
    url = "https://www.google.com.hk/search?q=%E7%BB%BF%E8%90%9D+%E7%BC%BA%E6%B0%B4+%E5%8F%B6%E7%89%87&safe=strict&hl=zh-CN&sxsrf=ALeKk02FYKs43o4HhBWZF81dQPEVA5Ywsw:1601955628573&source=lnms&tbm=isch&sa=X&ved=2ahUKEwiGkb_hhZ_sAhWTdd4KHX1hBWUQ_AUoAXoECAwQAw&biw=1440&bih=850"
    url = "https://image.baidu.com/search/index?tn=baiduimage&ps=1&ct=201326592&lm=-1&cl=2&nc=1&ie=utf-8&word=%E7%BB%BF%E8%90%9D+%E5%81%A5%E5%BA%B7+%E5%8F%B6%E5%AD%90"
    url = "https://image.baidu.com/search/index?tn=baiduimage&ipn=r&ct=201326592&cl=2&lm=-1&st=-1&fm=result&fr=&sf=1&fmq=1601954779511_R&pv=&ic=&nc=1&z=&hd=&latest=&copyright=&se=1&showtab=0&fb=0&width=&height=&face=0&istype=2&ie=utf-8&sid=&word=%E7%BB%BF%E8%90%9D+%E7%BC%BA%E6%B0%B4+%E5%8F%B6%E7%89%87"
    url = "https://image.baidu.com/search/index?tn=baiduimage&ipn=r&ct=201326592&cl=2&lm=-1&st=-1&fm=result&fr=&sf=1&fmq=1601963016840_R&pv=&ic=&nc=1&z=&hd=&latest=&copyright=&se=1&showtab=0&fb=0&width=&height=&face=0&istype=2&ie=utf-8&sid=&word=%E7%BB%BF%E8%90%9D+%E7%BC%BA%E5%85%89%E7%85%A7+%E5%8F%B6%E7%89%87"
    url = "https://image.baidu.com/search/index?tn=baiduimage&ipn=r&ct=201326592&cl=2&lm=-1&st=-1&fm=result&fr=&sf=1&fmq=1601964240381_R&pv=&ic=&nc=1&z=&hd=&latest=&copyright=&se=1&showtab=0&fb=0&width=&height=&face=0&istype=2&ie=utf-8&sid=&word=%E7%BB%BF%E8%90%9D+%E7%BC%BA%E5%85%89%E7%85%A7+%E5%8F%B6%E7%89%87+%E6%A0%B7%E5%AD%90"
    url = "https://image.baidu.com/search/index?tn=baiduimage&ipn=r&ct=201326592&cl=2&lm=-1&st=-1&fm=result&fr=&sf=1&fmq=1601969475839_R&pv=&ic=0&nc=1&z=0&hd=0&latest=0&copyright=0&se=1&showtab=0&fb=0&width=&height=&face=0&istype=2&ie=utf-8&sid=&word=%E7%BB%BF%E8%90%9D+%E6%99%92%E4%BC%A4+%E5%8F%B6%E7%89%87+%E6%A0%B7%E5%AD%90"
    url = "https://image.baidu.com/search/index?tn=baiduimage&ipn=r&ct=201326592&cl=2&lm=-1&st=-1&fm=result&fr=&sf=1&fmq=1601969475839_R&pv=&ic=0&nc=1&z=0&hd=0&latest=0&copyright=0&se=1&showtab=0&fb=0&width=&height=&face=0&istype=2&ie=utf-8&sid=&word=%E7%BB%BF%E8%90%9D+%E6%99%92%E4%BC%A4+%E5%8F%B6%E7%89%87+%E6%A0%B7%E5%AD%90"
    urls = []
    for i in params:
        urls.append(requests.get(url,headers = headers, params=i).json(strict=False).get('data'))

    return urls

def getImg(datalist, path):
    x = 0
    for list in datalist:
        for i in list:
            if i.get('thumbURL') != None:
                print('?????%s' % i.get('thumbURL'))
                urllib.request.urlretrieve(i.get('thumbURL'), path + "new_5_" +'%d.jpg' % x)
                x += 1
            else:
                print('???????')

if __name__ == '__main__':
    # headers = {"User-Agent": "User-Agent:Mozilla/5.0 (compatible; MSIE 9.0; Windows NT 6.1; Trident/5.0;"}
    save_pth = r'F:/piture_download/'
    headers = {"User-Agent":"User-Agent:Mozilla/5.0 (Macintosh; U; Intel Mac OS X 10_6_8; en-us) AppleWebKit/534.50 (KHTML, like Gecko) Version/5.1 Safari/534.50",}
    datalist = getDatas("?? ?? ?? ??", 20)
    getImg(datalist, save_pth)