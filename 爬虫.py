import requests
import re
url = "https://car.autohome.com.cn/pic/series/6548-1-p1.html"
response = requests.get(url)
txt = response.text
# print(txt)
#使用正则表达式来匹配
#src="//car3.autoimg.cn/cardfs/product/g26/M08/E1/8F/480x360_0_q95_c42_autohomecar__ChtlxWUG8ZWADMU5ACkd2JBZTRw064.jpg"
#开头是src,结尾是.jpg
ret = re.findall(r'src(.*?)jpg',txt)
for i in range(len(ret)):
    #'="//car2.autoimg.cn/cardfs/product/g25/M07/C7/B2/480x360_0_q95_c42_autohomecar__ChtliGT2k4KALE4JADnCfxgvUIQ981.'
    url = "https://"+ret[i].split("//")[1] +"jpg"
    ret[i] = url
# print(ret)
headers = {
    "authority": "car2.autoimg.cn",
    "method": "GET",
    "scheme": "https",
    "accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
    "accept-encoding": "gzip, deflate, br, zstd",
    "accept-language": "zh-CN,zh;q=0.9,en;q=0.8,en-GB;q=0.7",
    "cache-control": "no-cache",
    "pragma": "no-cache",
    "referer": "https://car.autohome.com.cn/",
    "sec-fetch-dest": "image",
    "sec-fetch-mode": "no-cors",
    "sec-fetch-site": "cross-site",
    "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36 Edg/127.0.0.0"
}

#https://car2.autoimg.cn/cardfs/product/g25/M07/C7/B2/480x360_0_q95_c42_autohomecar__ChtliGT2k4KALE4JADnCfxgvUIQ981.jpg
save_path ="C:\\Users\\CUGac\\PycharmProjects\\astar\\.venv\\Scripts\\U-Net\\deta"
for i in range(len(ret)):
    print(ret[i])
    response = requests.get(url = ret[i],headers=headers)
    #检测响应码
    if response.status_code == 200:
        print("正在下载第"+str(i)+"张图片")
        with open(save_path+"\\"+str(i)+".jpg","wb") as f:
            f.write(response.content)
    else:
        print(response.status_code)
print("爬取完成")