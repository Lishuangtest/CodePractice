# # 代码片段1
# import urllib.request
# import json
# import os
# import pathlib
# import re
# from bs4 import BeautifulSoup
#
# # 获取ID和name，并下载
#
#
# def json_txt(jsonSkinSJSON, defalut):
#     jsonSkinSJSON = jsonSkinSJSON["data"]["skins"]
#     i = 0
#     imgId = ""
#     imgName = ""
#     for key in jsonSkinSJSON:
#         if i == 0:
#             # print(key["id"])
#             imgId = key["id"]
#             # print(defalut)
#             imgName = defalut
#             i = i + 1
#         else:
#             imgId = key["id"]
#             imgName = key["name"]
#         save_dir = 'D:\LOLheroskin\\'
#         save_file_name = save_dir + imgName + ".jpg"
#         urlDown = "http://ossweb-img.qq.com/images/lol/web201310/skin/big" + imgId + ".jpg"
#         # print(urlDown)
#         try:
#             if not os.path.exists(save_file_name):
#                 urllib.request.urlretrieve(urlDown, save_file_name)
#         except Exception:
#             print("下载失败")
#
#
# # 获取英雄联盟皮肤
#
#
# def getSkins(urlOne):
#     response = urllib.request.urlopen(
#         urlOne)
#     data = response.read().decode('utf-8')
#     # print(data)
#     jsonSkin = re.findall(r"{\"data\":(.+?);", data)
#     jsonSkinS = "{\"data\":" + jsonSkin[0]
#
#     jsonSkinSJSON = json.loads(jsonSkinS)
#     # print(jsonSkinSJSON["data"]["name"])
#     defalut = jsonSkinSJSON["data"]["name"]
#
#     json_txt(jsonSkinSJSON, defalut)
#
# # 获取英雄联盟英雄列表
# response = urllib.request.urlopen(
#     "http://lol.qq.com/biz/hero/champion.js")
#
# data = response.read().decode('utf-8')
#
# json1 = re.findall(r"LOLherojs.champion=(.+?);", data)
#
# hero_json = json.loads(json1[0])['keys']
#
# c = []
# for key in hero_json:
#     # print("****key--：%s value--: %s" % (key, hero_json[key]))
#     url_skin = "http://lol.qq.com/biz/hero/" + hero_json[key] + ".js"
#     c.append(url_skin)
#
# # 文件夹不存在则创建
# save_dir = 'D:\LOLheroskin\\'
# if not os.path.exists(save_dir):
#     os.mkdir(save_dir)
#
# for heroOne in c:
#     getSkins(heroOne)
# print("下载完成")
import json
import re

import requests
import time
#获取JS源代码 获取英雄ID
#拼接URL地址
#获取图片下载地址
#下载图片

#驼峰命名法
#获取英雄图片
def getLOLImages():
    header = {'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/67.0.3396.62 Safari/537.36'}
    url_js = 'http://lol.qq.com/biz/hero/champion.js'
    #获取JS源代码 Str bytes
    res_js = requests.get(url_js).content
    #转码 转成字符串
    html_js = res_js.decode()
    #正则表达式
    req = '"keys":(.*?),"data"'
    list_js = re.findall(req,html_js)
    #转成dict
    dict_js = json.loads(list_js[0])
    # print(type(dict_js))
    #定义图片列表
    pic_list = []
    for key in dict_js:
        # print(key)#英雄ID
        #拼接URL
        for i in range(20):
            number = str(i)
            if len(number) == 1:
                hero_num = "00"+number
            elif len(number) == 2:
                hero_num = "0"+number
            numstr = key+hero_num
            url = "http://ossweb-img.qq.com/images/lol/web201310/skin/big"+numstr+".jpg"
            #http://ossweb-img.qq.com/images/lol/web201310/skin/big81000.jpg
            pic_list.append(url)
         #获取图片名称
        list_filepath = []
        path = "D:\Pycharmdaima\Pachong\LOLTU\\"
    for name in dict_js.values():
        for i in range(20):
            file_path = path+name+str(i)+'.jpg'
            list_filepath.append(file_path)
    #下载图片
    n = 0
    for picurl in pic_list:
        res = requests.get(picurl)
        n += 1
        #获取状态码
        if res.status_code == 200:
            print("正在下载%s"%list_filepath[n])
            time.sleep(1)
            with open(list_filepath[n],'wb') as f:
                f.write(res.content)
getLOLImages()
