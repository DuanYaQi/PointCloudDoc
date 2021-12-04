# Linux

## 命令

| 命令                          | 功能                                                         |
| :---------------------------- | :----------------------------------------------------------- |
| g++ -v                        | 查看版本                                                     |
| g++ **main.cpp**              | 编译c++文件                                                  |
| ./a.out                       | 运行可执行文件                                               |
| sudo apt-get install **xxx**  | 安装xxx                                                      |
| sudo apt-get update           | 更新源列表  /etc/apt/sources.list 读取软件列表               |
| sudo apt-get upgrade          | 把本地已安装的软件，与刚下载的软件列表里对应软件进行对比     |
| git clone **https://xxx.git** | 下载源码                                                     |
|                               |                                                              |
| sudo apt-get remove **xxx**   | 卸载掉xxx                                                    |
| sudo apt-get purge **xxx**    | 彻底卸载删除xxx的相关配置文件                                |
| sudo apt-get autoclean        | 清理安装软件时候留下的缓存程序软件包                         |
| sudo apt-get clean            | 清理安装软件时候留下的缓存程序软件包                         |
| sudo apt-get autoremove       | 卸载不需要的依赖关系   /一般不用                             |
|                               |                                                              |
| sudo dpkg -i xxx              | 安装xxx软件  .deb                                            |
| sudo dpkg -l xxx              | 查看已经安装的软件                                           |
| sudo dpkg -r **xxx**          | 卸载xxx                                                      |
|                               |                                                              |
| locate **xxx**                | 一般寻找**文件**xxx 数据搜寻 搜索数据库（/var/lib/locatedb）<br>比find -name快。使用前，先使用**updatedb**命令，手动更新数据库 |
| find <指定目录> -name **xxx** | 一般寻找**文件**xxx 硬盘搜寻 档案档名的搜寻  默认当前目录    |
| whereis **xxx**               | 一般寻找**命令**xxx  程序名的搜索，默认全部搜索<br>二进制文件（参数-b）、man说明文件（参数-m）和源代码文件（参数-s） |
| which **xxx**                 | 在PATH变量指定的路径中，搜索系统**命令**xxx的位置，<br>并返回第一个搜索结果。<br>可以看到某个系统命令是否存在，以及执行的到底是哪一个位置的命令 |
|                               |                                                              |
|                               |                                                              |
| tar -zxvf **xxx.tar.gz**      | 解压 xxx.tar.gz                                              |
| chmod +x **xxx**              | 给文件xxx加上x可执行权限                                     |
| chmod 777 **xxx**             | 文件xxx给所有用户全部权限   r:4     w:2    x:1               |
|                               |                                                              |
| ps -aux\|grep **xxx**         | 查看xxx的相关进程信息                                        |
| kill **xxx**                  | 关闭pid=xxx的进程                                            |
|                               |                                                              |
| mkdir -p **xxx***             | 建立多层目录                                                 |
| ls -a -l =ll                  | -a连同隐藏档(开头.)一起列出来、-l包括属性权限                |
| cp [-irpa] **xxx** **yyy**    | -i 覆盖询问 -r 递归复制目录 -p 档案属性一起复制<br/> -d 复制连结文件属性而非档案本身  -a=-pdr  xxx为来源档  yyy为目的档 |
| rm [-rf] **xxx**              | -f 强制 -r递归删除 xxx待删除                                 |
| mv [-f] **xxx** **yyy**       | -f强制覆盖  xxx为来源档  yyy为目的档  也可用来改名           |
| cat [-n] **xxx**              | -n打印行号  打印xxx内容                                      |
| file **xx**                   | 查看文件类型 ASCII data binary                               |
| nano **xxx.txt**              | 存在就开启旧档，不存在就开启新档 类似vi/vim                  |
|                               |                                                              |
| ssh -l root **xxx**           | 连接IPxxx的远程服务器                                        |
|                               |                                                              |
|                               |                                                              |
| dpkg -l \| grep ssh           | grep 过滤不包含ssh的  global search regularexpression        |
| ps -e \| grep ssh             |                                                              |
|                               |                                                              |
|                               |                                                              |
|                               |                                                              |
|                               |                                                              |
|                               |                                                              |
|                               |                                                              |
|                               |                                                              |
|                               |                                                              |











## 快捷键

| 快捷键                       | 功能                                   |
| ---------------------------- | :------------------------------------- |
| ctrl+alt+t                   | 打开终端                               |
| Shift+Ctrl+t                 | 在终端中创建新标签                     |
| Shift+Ctrl+w                 | 关闭当前新标签                         |
| Shift+Ctrl+Q                 | 关闭终端                               |
| ctrl + super + d             | 最小化所有窗口                         |
| alt + f4                     | 关闭当前窗口                           |
| alt + `                      | 切换当前运行的应用程序窗口             |
| ctrl + insert                | 命令行复制                             |
| shift + insert               | 命令行粘贴                             |
| Shift+Ctrl+C                 | 复制                                   |
| Shift+Ctrl+V                 | 粘贴                                   |
| ctrl + l                     | 清屏                                   |
| ctrl + a                     | 光标行首                               |
| ctrl + e                     | 光标行尾                               |
| Ctrl + u                     | 擦除从当前光标位置到行首的全部内容     |
| Ctrl + k                     | 擦除的是从当前光标位置到行尾的全部内容 |
|                              |                                        |
| [Ctrl] + [Alt] + [F1] ~ [F6] | 文字接口登入 tty1 ~ tty6 终端机        |
| [Ctrl] + [Alt] + [F7]        | 图形接口桌面                           |
| [Ctrl] + [h]                 | 显示隐藏文件路径                       |
|                              |                                        |
|                              |                                        |
|                              |                                        |
|                              |                                        |
|                              |                                        |
|                              |                                        |
|                              |                                        |
|                              |                                        |
|                              |                                        |
|                              |                                        |



## 网站



| 网站                                      | 功能                 |
| :---------------------------------------- | -------------------- |
| http://zhoudaxiaa.gitee.io/downgit/#/home | 下载github单个文件   |
| https://wallpaperhub.app/                 | 壁纸下载             |
| https://openslam.org/                     | OpenSLAM             |
| http://www.cvlibs.net/datasets/kitti/     | simulation-Kitti图库 |
|                                           |                      |
|                                           |                      |
|                                           |                      |
|                                           |                      |
|                                           |                      |





## 常见问题

### /boot空间不足

https://blog.csdn.net/along_oneday/article/details/75148240
https://www.linuxidc.com/Linux/2017-12/149655.htm





### /home空间不足

```
在磁盘里查看
```





### nvidia驱动 核显

https://blog.csdn.net/xiaokedou_hust/article/details/82187860



### vpn安装

https://segmentfault.com/a/1190000010533832



### 美化ubuntu

https://zhuanlan.zhihu.com/p/63584709?utm_source=qq&utm_medium=social&utm_oi=730754962790813696



---

# SSH

https://blog.csdn.net/li528405176/article/details/82810342

如果只是想远程登陆别的机器只需要安装客户端（Ubuntu默认安装了客户端），如果要开放本机的SSH服务就需要安装服务器。

```
sudo apt-get install openssh-client 
sudo apt-get install openssh-server
```







免密登录

**本地客户端生成公私钥**

```shell
ssh-keygen
~/.ssh
id_rsa私钥
id_rsa.pub公钥
```



**公钥上传至服务器**

```shell
ssh-copy-id -i ~/.ssh/id_rsa.pub root@192.168.235.22
ssh-copy-id -i ~/.ssh/id_rsa.pub -p 37490 root@region-4.autodl.com
```

-p为端口号



```shell
Now try logging into the machine, with:   "ssh -p '37490' 'root@region-4.autodl.com'"
and check to make sure that only the key(s) you wanted were added.
```





---

# Vim

## 命令模式

i切换到**输入模式**，已输入字符

x删除当前光标所在的字符

:切换到底线命令模式

yy复制当前行、p粘贴到下一行、dd删除当前行 

nyy复制n行、ndd删除n行 、np复制多遍



## 输入模式

home/end 行首行尾

page up/down上下翻页



## 底线命令模式

q退出 w保存  !非

:0  第一行 

:$  最后一行

/find 寻找带有find字样的内容









---

# 双系统安装 Ubuntu18.04LTS + Windows

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

删除无用软件

删除libreoffice

```shell
sudo apt-get remove libreoffice-common  
```

删除Amazon的链接

```shell
sudo apt-get remove unity-webapps-common  
```

删掉基本不用的自带软件（用的时候再装也来得及）

```shell
sudo apt-get remove thunderbird totem rhythmbox empathy brasero simple-scan gnome-mahjongg aisleriot gnome-mines cheese transmission-common gnome-orca webbrowser-app gnome-sudoku  landscape-client-ui-install

sudo apt-get remove onboard deja-dup  
```





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





### 安装anydesk

步骤1.首先，通过apt在终端中运行以下命令来确保所有系统软件包都是最新的。

```shell
sudo apt update
sudo apt upgrade
```

步骤2.在Ubuntu 20.04上安装AnyDesk。

现在，我们将存储库密钥添加到“受信任的软件提供者”列表中，然后通过运行以下命令将PPA添加到您的系统中：

```shell
wget -qO - https://keys.anydesk.com/repos/DEB-GPG-KEY | sudo apt-key add - 
sudo echo "deb http://deb.anydesk.com/ all main" > /etc/apt/sources.list.d/anydesk.list
```

然后，运行以下命令从存储库中安装Anydesk及其依赖项：

```shell
sudo apt update 
sudo apt install anydesk
```

步骤3.在Ubuntu系统上访问AnyDesk。

成功安装后，您可以通过在应用程序启动器中键入Anydesk来启动。现在，您可能需要为无人参与访问设置密码。这将在您的系统上设置一个固定的密码，该密码可随时用于连接。