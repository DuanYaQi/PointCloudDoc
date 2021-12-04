# TikTok





## TiKok-US.conf

```conf
[URL Rewrite]
(?<=_region=)CN(?=&) US 307
(?<=&mcc_mnc=)4 2 307
^(https?:\/\/(tnc|dm)[\w-]+\.\w+\.com\/.+)(\?)(.+) 13 302
(^https?:\/\/*\.\w{4}okv.com\/.+&.+)(\d{2}\.3\.\d)(.+) 118.03 302

[MITM]
hostname = *.tiktokv.com,*.byteoversea.com,*.tik-tokapi.com
```



## TikTok.list

```LIST
# NAME: TikTok

DOMAIN,p16-tiktokcdn-com.akamaized.net
DOMAIN-SUFFIX,byteoversea.com
DOMAIN-SUFFIX,ibytedtos.com
DOMAIN-SUFFIX,ibyteimg.com
DOMAIN-SUFFIX,ipstatp.com
DOMAIN-SUFFIX,muscdn.com
DOMAIN-SUFFIX,musical.ly
DOMAIN-SUFFIX,sgpstatp.com
DOMAIN-SUFFIX,snssdk.com
DOMAIN-SUFFIX,tik-tokapi.com
DOMAIN-SUFFIX,tiktok.com
DOMAIN-SUFFIX,tiktokcdn.com
DOMAIN-SUFFIX,tiktokv.com
DOMAIN-KEYWORD,-tiktokcdn-com
USER-AGENT,TikTok*
```



关于分流的规则
点选刚下载的配置文件，选择编辑纯文本
在[General]配置结束后添加保存

```list
[Rule]
# NAME: TikTok
DOMAIN,p16-tiktokcdn-com.akamaized.net,PROXY
DOMAIN-SUFFIX,amemv.com,PROXY
DOMAIN-SUFFIX,byteoversea.com,PROXY
DOMAIN-SUFFIX,ibytedtos.com,PROXY
DOMAIN-SUFFIX,ibyteimg.com,PROXY
DOMAIN-SUFFIX,ipstatp.com,PROXY
DOMAIN-SUFFIX,muscdn.com,PROXY
DOMAIN-SUFFIX,musical.ly,PROXY
DOMAIN-SUFFIX,sgpstatp.com,PROXY
DOMAIN-SUFFIX,snssdk.com,PROXY
DOMAIN-SUFFIX,tik-tokapi.com,PROXY
DOMAIN-SUFFIX,tiktok.com,PROXY
DOMAIN-SUFFIX,tiktokcdn.com,PROXY
DOMAIN-SUFFIX,tiktokv.com,PROXY
DOMAIN-KEYWORD,-tiktokcdn-com,PROXY
USER-AGENT,TikTok*,PROXY
```

