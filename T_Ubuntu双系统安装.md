# Ubuntu18.04LTS + Windows双系统安装

## 下载内容

1. ubuntu-18.04.5-desktop-amd64.iso  

   https://ubuntu.com/download/alternative-downloads

   http://mirrors.ustc.edu.cn/ubuntu-releases/18.04.5/

   http://mirrors.aliyun.com/ubuntu-releases/18.04.5/

2. 



## 备份数据





## 设置BIOS模式

​	"win+r"快捷键进入"运行"，输入"msinfo32"回车，出现以下界面，可查看BIOS模式，最好为UEFI模式

​	 部分电脑在BIOS中，打开关闭Security Boot即可





## U盘制作启动盘





## 优化

### vpn

shadowsocks-qt5   https://www.cnblogs.com/cpl9412290130/p/11814334.html



### typora

官网



### zsh

安装插件`highlight`，高亮语法

https://blog.csdn.net/ice__snow/article/details/80152068



### 主题

主题flatabulous/ant  https://www.opendesktop.org/p/1099856/

图标ultra-flat



### 自动挂载硬盘

https://www.cnblogs.com/zifeiy/p/9142086.html

https://blog.csdn.net/u014436243/article/details/89952671



### 关闭grub

https://blog.csdn.net/ice__snow/article/details/80152068#t28

```bash
sudo vi /etc/default/grub
#GRUB_DEFAULT=0
#GRUB_TIMEOUT_STYLE=hidden
#GRUB_TIMEOUT=10
#GRUB_DISTRIBUTOR=`lsb_release -i -s 2> /dev/null || echo Debian`
#GRUB_CMDLINE_LINUX_DEFAULT="quiet splash"
#GRUB_CMDLINE_LINUX=""

替换为
GRUB_DEFAULT=0
GRUB_HIDDEN_TIMEOUT=0
GRUB_HIDDEN_TIMEOUT_QUIET=true
GRUB_TIMEOUT=0
GRUB_DISTRIBUTOR=`lsb_release -i -s 2> /dev/null || echo Debian`
GRUB_CMDLINE_LINUX_DEFAULT="quiet splash"
GRUB_CMDLINE_LINUX=""
GRUB_DISABLE_OS_PROBER=true
```



### 安装copytranslator

https://copytranslator.github.io/download/linux.html#v9-%E5%AF%92%E6%B8%90

```bash
sudo dpkg -i copytranslator_9.1.0_amd64.deb
```



### xdm

卸载https://www.kutu66.com/ubuntu/article_166960



### 安装openGL

https://blog.csdn.net/renhaofan/article/details/82631082



### 设置python3.6为python

```bash
sudo rm /usr/bin/python
sudo ln -sf /usr/bin/python3.6 /usr/bin/python
```





### ubuntu 18.04设置开机自动挂载移动硬盘

https://www.cnblogs.com/zifeiy/p/9142086.html

https://blog.csdn.net/u014436243/article/details/89952671

