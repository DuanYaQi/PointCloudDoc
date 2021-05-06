# E_Flow

```shell
docker run --runtime=nvidia --rm -it -w /home -v /home/duan/windows/udata:/home/data/ -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=unix$DISPLAY -e GDK_SCALE -e GDK_DPI_SCALE -p 127.0.0.2:8097:8097 nvidia/cudagl:duan

docker commit -p 12f061bdad7f nvidia/cudagl:duan

watch -n 0.1 -d nvidia-smi

tensorboard --logdir ./log --port 8890

然后进行端口映射就行了
```







https://deep-generative-models.github.io/index2020.html

https://www.pianshen.com/article/27411637207/

http://www.bubuko.com/infodetail-3317421.html

https://zhuanlan.zhihu.com/p/59615785

https://zhuanlan.zhihu.com/p/305627568

https://www.zhihu.com/question/376122890/answer/1399139778





