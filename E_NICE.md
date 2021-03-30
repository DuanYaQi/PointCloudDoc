# E_NICE

```shell
docker run --runtime=nvidia --rm -it -w /home -v /home/duan/windows/udata:/home/data/ -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=unix$DISPLAY -e GDK_SCALE -e GDK_DPI_SCALE -p 127.0.0.2:8097:8097 nvidia/cudagl:duan

docker commit -p 19f8e2ee0850 nvidia/cudagl:duan
```

