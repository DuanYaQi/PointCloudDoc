# pypoisson





## error

1. src/pypoisson.cpp:4:20: fatal error: Python.h: No such file or directory

```bash
sudo apt-get install python3.6-dev

------------------------------------------------------------
sudo apt-get install python-dev   # for python2.x installs
sudo apt-get install python3-dev  # for python3.x installs
```



```bash
old
libwayland-egl.so.1 (libc6,x86-64) => /usr/lib/x86_64-linux-gnu/libwayland-egl.so.1
	libwayland-egl.so (libc6,x86-64) => /usr/lib/x86_64-linux-gnu/libwayland-egl.so
	libcogl.so.20 (libc6,x86-64) => /usr/lib/x86_64-linux-gnu/libcogl.so.20
	libOpenGL.so.0 (libc6,x86-64) => /usr/lib/x86_64-linux-gnu/libOpenGL.so.0
	libOpenGL.so (libc6,x86-64) => /usr/lib/x86_64-linux-gnu/libOpenGL.so
	libGL.so.1 (libc6,x86-64) => /usr/lib/x86_64-linux-gnu/libGL.so.1
	libGL.so.1 (libc6) => /usr/lib/i386-linux-gnu/libGL.so.1
	libGL.so (libc6,x86-64) => /usr/lib/x86_64-linux-gnu/libGL.so
	libEGL.so.1 (libc6,x86-64) => /usr/lib/x86_64-linux-gnu/libEGL.so.1
	libEGL.so (libc6,x86-64) => /usr/lib/x86_64-linux-gnu/libEGL.so

```



```bash
new
libwayland-egl.so.1 (libc6,x86-64) => /usr/lib/x86_64-linux-gnu/libwayland-egl.so.1
	libwayland-egl.so (libc6,x86-64) => /usr/lib/x86_64-linux-gnu/libwayland-egl.so
	libcogl.so.20 (libc6,x86-64) => /usr/lib/x86_64-linux-gnu/libcogl.so.20
	libOpenGL.so.0 (libc6,x86-64) => /usr/lib/x86_64-linux-gnu/libOpenGL.so.0
	libOpenGL.so (libc6,x86-64) => /usr/lib/x86_64-linux-gnu/libOpenGL.so
	libGL.so.1 (libc6,x86-64) => /usr/lib/x86_64-linux-gnu/libGL.so.1
	libGL.so.1 (libc6) => /usr/lib/i386-linux-gnu/libGL.so.1
	libGL.so (libc6,x86-64) => /usr/lib/x86_64-linux-gnu/libGL.so
	libEGL.so.1 (libc6,x86-64) => /usr/lib/x86_64-linux-gnu/libEGL.so.1
	libEGL.so (libc6,x86-64) => /usr/lib/x86_64-linux-gnu/libEGL.so

```



```
new in docker
libwayland-egl.so.1 (libc6,x86-64) => /usr/lib/x86_64-linux-gnu/libwayland-egl.so.1
	libwayland-egl.so (libc6,x86-64) => /usr/lib/x86_64-linux-gnu/libwayland-egl.so
	libGL.so.1 (libc6,x86-64) => /usr/lib/x86_64-linux-gnu/mesa/libGL.so.1
	libGL.so (libc6,x86-64) => /usr/lib/x86_64-linux-gnu/libGL.so
	libGL.so (libc6,x86-64) => /usr/lib/x86_64-linux-gnu/mesa/libGL.so
	libEGL.so.1 (libc6,x86-64) => /usr/lib/x86_64-linux-gnu/mesa-egl/libEGL.so.1
	libEGL.so (libc6,x86-64) => /usr/lib/x86_64-linux-gnu/libEGL.so
	libEGL.so (libc6,x86-64) => /usr/lib/x86_64-linux-gnu/mesa-egl/libEGL.so

```

