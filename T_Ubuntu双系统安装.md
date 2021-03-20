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

​	1.备份你U 盘；

​	2.进入软碟通，选择文件，浏览到ubuntu镜像所在的目录，选择ubuntu镜像文件，双击打开

​	3.菜单栏选择"启动"，选择"写入硬盘映像"

​	4.选择格式化，选择写入



## Windows下创建空间分区

### 历史分区配置

总空间extended 20G

/boot 976M     已用257.7M 未用718.2M

/ 28.61G           已用20.3G    未用8.2G

/home 82.81G  已用71.9G  未用11.5G

linux-swap        7.36G



现在分200G给ubuntu系统 保持未分配状态



逻辑分区

/boot 1Gb   

/swap 7Gb  

/ 分区 50Gb

/home 90Gb





### 安装过程

Windows下安装Ubuntu 16.04双系统

https://www.cnblogs.com/Duane/p/5424218.html

Win10 Ubuntu16.04/Ubuntu18.04双系统完美安装

https://blog.csdn.net/qq_24624539/article/details/81775635

​	为图形或无线硬件，以及MP3和其它媒体安装第三方软件

​	其他选项

​	正常安装

​	

​	发现有空闲分区

​	



最后bcd用2.3并且grub引导不要用grub2



---

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

