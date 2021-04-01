## HelloWorld-LV.1

​	main.c

```c++
#include<stdio.h>
int main(){
    printf("Hello World from t1 Main!\n");
    return 0;
}
```

​	CMakeList.txt

```cmake
PROJECT(HELLO)  
#该指令指定定义工程名称，并指定工程支持的语言 可以忽略，默认表示支持所有语言PROJECT(projectname [CXX] [C] [Java])
#这个指令隐式的定义了两个cmake 变量:<projectname>_BINARY_DIR 以及<projectname>_SOURCE_DIR
#这里就是HELLO_BINARY_DIR 和 HELLO_SOURCE_DIR直接使用PROJECT_BINARY_DIR，PROJECT_SOURCE_DIR即可

SET(SRC_LIST main.c)	
#显式的定义变量（与隐式对应） 如果有多个源文件，也可以定义成：SET(SRC_LIST main.c t1.c t2.c)

MESSAGE(STATUS "This is BINARY dir "${HELLO_BINARY_DIR})  
MESSAGE(STATUS "This is SOURCE dir "${HELLO_SOURCE_DIR})
#输出消息 SEND_ERROR 产生错误，生成过程被跳过
#SATUS 输出前缀为-的信息
#FATAL_ERROR 立即终止所有cmake过程

ADD_EXECUTABLE(hello ${SRC_LIST})     
#会生成一个名为hello的可执行文件，相关源文件是SRC_LIST中定义的源文件列表
#也可以直接写成 ADD_EXECUTABLE(hello main.c) 
```

---

## HelloWorld-LV.2

1. src/main.c

   ```c++
   #include<stdio.h>
   
   int main(){
   	printf("Hello World from t1 Main!\n");
   	return 0;
   }
   ```

2. src/CMakeLists

   ```cmake
   ADD_EXECUTABLE(hello main.c)
   INSTALL(TARGETS hello RUNTIME DESTINATION bin) #目标可执行文件放置在bin目录下
   ```

3. 主目录下CMakeLists

   ```cmake
   PROJECT(HELLO)
   ADD_SUBDIRECTORY(src bin)
   ##ADD_SUBDIRECTORY(source_dir [binary_dir] [EXCLUDE_FROM_ALL])
   #这个指令用于向当前工程添加存放源文件的子目录，并可以指定中间二进制和目标二进制存放的位置。
   
   #上面的例子定义了将src子目录加入工程，并指定编译输出(包含编译中间结果)路径为bin目录。
   #如果不进行bin目录的指定，那么编译结果(包括中间结果)都将存放在build/src目录(这个目录跟原有的src目录对应)，指定bin目录后，相当于在编译时将src重命名为bin，所有的中间结果和目标二进制都将存放在bin目录。
   
   INSTALL(FILES COPYRIGHT README DESTINATION share/doc/cmake/t2)#readme普通文件
   INSTALL(PROGRAMS runhello.sh DESTINATION bin)			#.sh非目标的可执行文件
   INSTALL(DIRECTORY doc/ DESTINATION share/doc/cmake/t2) #doc目录
   ```

4. 添加一个子目录 doc，用来放置这个工程的文档 hello.txt

5. 在工程目录添加文本文件 COPYRIGHT, README

6. 在工程目录添加一个 runhello.sh 脚本，用来调用 hello 二进制

7. 将构建后的目标文件放入构建目录的 bin 子目录

8. 最终安装这些文件：将 hello 二进制与 runhello.sh 安装至`/usr/bin`，将 doc 目录的内容以及 COPYRIGHT/README 安装到`/usr/share/doc/cmake/t2`

9. 建立build目录进行外部编译 目标文件hello在`build/bin`目录中

   安装有两种：从代码编译后直接`make install`安装，另一种是打包时的指定目录安装

   ```shell
   cmake -DCMAKE_INSTALL_PREFIX=~/Workspace/Learning-Slam ..
   #CMAKE_INSTALL_PREFIX默认/usr/local
   make
   make install
   
   #构建失败，如果需要查看细节，可以使用第一节提到的方法
   make VERBOSE=1 #来构建
   ```

---

## HelloWorld-LV.3

１. 建立一个静态库和动态库，提供 HelloFunc 函数供其他程序编程使用，HelloFunc 向终端输出 Hello World 字符串。
２. 安装头文件与共享库。

目录下CMakeLists.txt

```cmake
PROJECT(HELLOLIB)
ADD_SUBDIRECTORY(lib)
```

lib/hello.c

```c
#include “hello.h”
void HelloFunc()
{
	printf(“Hello World\n”);
}
```

lib/hello.h

```h
#ifndef HELLO_H
#define HELLO_H
#include <stdio.h>
void HelloFunc();
#endif
```

lib/CMakeLists.txt

```cmake
SET(LIBHELLO_SRC hello.c)
ADD_LIBRARY(hello SHARED ${LIBHELLOSRC})
ADD_LIBRARY(hello_static STATIC ${LIBHELLO_SRC})

SET_TARGET_PROPERTIES(hello_static PROPERTIES OUTPUT_NAME "hello")#输出名控制
SET_TARGET_PROPERTIES(hello PROPERTIES VERSION 1.2 SOVERSION 1)#动态库版本

INSTALL(TARGETS hello hello_static LIBRARY DESTINATION lib ARCHIVE DESTINATION lib)
INSTALL(FILES hello.h DESTINATION include/hello)
```

src/main.c

```c
#include<hello.h>
int main(){
	HelloFunc();
	return 0;
}
```

src/CMakeLists.txt

```cmake
ADD_EXECUTABLE(main main.c)
INCLUDE_DIRECTORIES(~/Workspace/Learning-Slam/include/hello)#头文件搜索
TARGET_LINK_LIBRARIES(main libhello.a)
TARGET_LINK_LIBRARIES(main libhello.so)
```

---

## 语法

1. 规则

   ```cmake
   #1.变量使用${}方式取值，但是在IF控制语句中直接使用变量名
   #2.指令(参数1 参数2)参数使用括弧括起，参数之间使用空格或分号分开
   ADD_EXECUTABLE(hello main.c;func.c)
   #3.指令大小写无关，参数和变量大小写相关。但推荐大写指令
   #4.工程名的HELLO和可执行文件hello是毫无关系的 也可写成
   ADD_EXECUTABLE(t1 main.c) 
   #5.清理工程构建结果即可执行文件 make clean
   #6.外部构建在build文件夹下进行 内部构建直接在主路径下
   ```

2. 换个地方保存二进制

   ```cmake
   #指定最终目标二进制的位置(最终生成的hello或者最终的共享库，不包含编译生成的中间文件)
   PROJECT_BINARY_DIR外部编译目录build
   SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin) 
   SET(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)
   #可执行二进制的输出路径为 build/bin 和库的输出路径为 build/lib 
   #在哪里ADD_EXECUTABLE 或 ADD_LIBRARY，如果需要改变目标存放路径，就在哪里加入上述的定义
   ```

---

## 安装

- 代码编译后`make install`安装
- 打包时制定目录安装

Makefile

```makefile
DESTDIR=
install:
	mkdir -p $(DESTDIR)/usr/bin
	install -m 755 hello $(DESTDIR)/usr/bin
```

可以通过`make install`将hello直接安装到`/usr/bin`目录。也可以通过`make install DESTDIR=/tmp/test`将hello安装在`/tmp/test/usr/bin`目录。

---

稍微复杂一点的需要定义PREFIX，一般autotools工程，会运行这样的指令:

`./configure -prefix=/usr`或者`./configure --prefix=/usr/local`来指定PREFIX

Makefile改写为

```makefile
DESTDIR=
PREFIX=/usr
install:
	mkdir -p $(DESTDIR)/$(PREFIX)/bin
	install -m 755 hello $(DESTDIR)/$(PREFIX)/bin
```

HelloWorld安装方法：需要引入变量`CMAKE_INSTALL_PREFIX`，该变量类似configure脚本的-prefix，

`cmake -DCMAKE_INSTALL_PREFIX=/usr`

INSTALL指令用于定义安装规则，安装的内容可以包括目标二进制、动态库、静态库以及文件、目录、脚本等。

### **目标文件的安装**

```cmake
INSTALL(TARGETS targets...
	[[ARCHIVE|LIBRARY|RUNTIME]
 		[DESTINATION <dir>]
 		[PERMISSIONS permissions...]
 		[CONFIGURATIONS [Debug|Release|...]]
 		[COMPONENT <component>]
 		[OPTIONAL]
 	 	] [...])

#参数后的target后面跟的就是我们通过ADD_EXECUTABLE或者ADD_LIBRARY定义的目标文件，可能是执行二进制、动态库、静态库
#目标类型也就对应三种，ARCHIVE静态库，LIBRARY动态库，RUNTIME可执行目标二进制
```

```cmake
#DESTINATION定义了安装的路径，如果路径以/开头，绝对路径，这时候CMAKE_INSTALL_PREFIX无效。如果用相对路径，不要以/开头，安装后路径就是
${CMAKE_INSTALL_PREFIX}/<DESTINATION定义的路径>
INSTALL(TARGETS myrun mylib mystaticlib
	RUNTIME DESTINATION bin
	LIBRARY DESTINATION lib
	ARCHIVE DESTINATION libstatic
	)
#上面的例子会将
#可执行二进制 myrun		  安装到${CMAKE_INSTALL_PREFIX}/bin 目录
#动态库 libmylib 			安装到${CMAKE_INSTALL_PREFIX}/lib 目录
#静态库 libmystaticlib 	安装到${CMAKE_INSTALL_PREFIX}/libstatic 目录
```

### **普通文件的安装**

```cmake
INSTALL(FILES files... DESTINATION <dir>
	[PERMISSIONS permissions...]
	[CONFIGURATIONS [Debug|Release|...]] 
	[COMPONENT <component>]
	[RENAME <name>] [OPTIONAL])

#可用于安装一般文件，并可以指定访问权限，文件名是此指令所在路径下的相对路径，如果默认不定义权限安装后的权限为：644        OWNER_WRITE, OWNER_READ, GROUP_READ,和 WORLD_READ
```

### **非目标文件的可执行程序安装（脚本）**

```cmake
INSTALL(PROGRAMS files... DESTINATION <dir>
	[PERMISSIONS permissions...]
	[CONFIGURATIONS [Debug|Release|...]]
	[COMPONENT <component>]
	[RENAME <name>] [OPTIONAL])

#跟上面files指令一样，安装后权限为755     OWNER_WRITE, OWNER_READ, OWNER_EXECUTE；GROUP_READ, GROUP_EXECUTE 和 WORLD_READ, WORLD_EXECUTE
```

### **目录的安装**

```cmake
INSTALL(DIRECTORY dirs... DESTINATION <dir>
	[FILE_PERMISSIONS permissions...]
	[DIRECTORY_PERMISSIONS permissions...]
	[USE_SOURCE_PERMISSIONS]
	[CONFIGURATIONS [Debug|Release|...]]
	[COMPONENT <component>]
	[[PATTERN <pattern> | REGEX <regex>]
	 [EXCLUDE] [PERMISSIONS permissions...]] [...])

#DIRECTORY后面连接的是所在Source目录的相对路径，但务必注意：abc 和 abc/有很大的区别。
#如果目录名不以/结尾，那么这个目录将被安装为目标路径下的abc，如果目录名以/结尾，代表将这个目录中的内容安装到目标路径，但不包括这个目录本身。
#PATTERN 用于使用正则表达式进行过滤
#PERMISSIONS 用于指定 PATTERN 过滤后的文件权限。

INSTALL(DIRECTORY icons scripts/ DESTINATION share/myproj
	PATTERN "CVS" EXCLUDE
	PATTERN "scripts/*"
	PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ GROUP_EXECUTE GROUP_READ)

#执行结果是：
#将 icons 目录安装到 <prefix>/share/myproj
#将 scripts/中的内容安装到<prefix>/share/myproj
#不包含目录名为CVS的目录，对于scripts/*文件指定权限为 OWNER_EXECUTE OWNER_WRITE OWNER_READ GROUP_EXECUTE GROUP_READ.
```

### **安装时CMAKE脚本的执行**

```cmake
INSTALL([[SCRIPT <file>] [CODE <code>]] [...])
#SCRIPT 参数用于在安装时调用 cmake 脚本文件（也就是<abc>.cmake 文件）
#CODE 参数用于执行 CMAKE 指令，必须以双引号括起来。比如：
INSTALL(CODE "MESSAGE(\"Sample install message.\")")
```

---

## 静态库与动态库

1. **编译共享库**

```cmake
ADD_LIBRARY(libname [SHARED|STATIC|MODULE]
	[EXCLUDE_FROM_ALL] 
	source1 source2 ... sourceN)

#EXCLUDE_FROM_ALL 参数的意思是这个库不会被默认构建，除非有其他的组件依赖或者手工构建。
```

你不需要写全 libhello.so，只需要填写 hello 即可，cmake 系统会自动为你生成libhello.X

类型有三种： SHARED动态库、STATIC静态库、MODULE

2. **添加静态库**

按照一般的习惯，静态库名字跟动态库名字应该是一致的，只不过后缀是.a 罢了。下面我们用这个指令再来添加静态库：

```cmake
ADD_LIBRARY(hello STATIC ${LIBHELLO_SRC})
#然后再在 build 目录进行外部编译，我们会发现，静态库根本没有被构建，仍然只生成了一个动态库。因为 hello 作为一个 target 是不能重名的，所以，静态库构建指令无效。
#如果我们把上面的 hello 修改为 hello_static,就可以构建一个 libhello_static.a 的静态库了。
ADD_LIBRARY(hello_static STATIC ${LIBHELLO_SRC})
```

这种结果显示不是我们想要的，我们需要的是名字相同的静态库和动态库，因为 target 名称是唯一的，所以，我们肯定不能通过 ADD_LIBRARY 指令来实现了。这时候我们需要用到另外一个指令

```cmake
SET_TARGET_PROPERTIES(target1 target2 ...
	PROPERTIES prop1 value1
	prop2 value2 ...)

#这条指令可以用来设置输出的名称，对于动态库，还可以用来指定动态库版本和 API 版本。
#在本例中，我们需要作的是向 lib/CMakeLists.txt 中添加一条：
SET_TARGET_PROPERTIES(hello_static PROPERTIES OUTPUT_NAME "hello")
#这样，我们就可以同时得到 libhello.so/libhello.a 两个库
```

3. **动态库版本号**

   ```cmake
   SET_TARGET_PROPERTIES(hello PROPERTIES VERSION 1.2 SOVERSION 1)
   #VERSION 指代动态库版本，SOVERSION 指代 API 版本。
   ```

4. **安装共享库和头文件**

   ```cmake
   INSTALL(TARGETS hello hello_static			#
   	LIBRARY DESTINATION lib                 #.so
   	ARCHIVE DESTINATION lib)				#.a
   INSTALL(FILES hello.h DESTINATION include/hello) #.h
   ```

---

## 使用外部共享库和头文件

1. 引入头文件搜索路径

   ```cmake
   #为了让我们的工程能够找到 hello.h 头文件，我们需要引入一个新的指令
   INCLUDE_DIRECTORIES([AFTER|BEFORE] [SYSTEM] dir1 dir2 ...)
   #这条指令可以用来向工程添加多个特定的头文件搜索路径，路径之间用空格分割，如果路径中包含了空格，可以使用双引号将它括起来，默认的行为是追加到当前的头文件搜索路径的后面
   
   #现在我们在 src/CMakeLists.txt 中添加一个头文件搜索路径，方式很简单，加入：
   INCLUDE_DIRECTORIES(/usr/include/hello)
   ```

2. 为target添加共享库

   ```cmake
   LINK_DIRECTORIES(directory1 directory2 ...)
   #这个指令非常简单，添加非标准的共享库搜索路径，比如，在工程内部同时存在共享库和可执行二进制，在编译时就需要指定一下这些共享库的路径。
   
   TARGET_LINK_LIBRARIES(target library1
   	<debug | optimized> library2 
   	...)
   #这个指令可以用来为 target 添加需要链接的共享库，本例中是一个可执行文件，但是同样可以用于为自己编写的共享库添加共享库链接。
   ```

   

---