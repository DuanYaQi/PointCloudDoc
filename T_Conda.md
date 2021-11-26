# Conda



## 下载安装

```bash
下载安装
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.s
bash Miniconda3-latest-Linux-x86_64.sh
```



## 配置包下载源

```shell
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/bioconda/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/
conda config --set show_channel_urls yes


conda env create -f environment.yaml
conda activate sap
```





---

## 环境

每个环境都有自己独立的软件或开发包列表，并会自动添加相应的环境变量和依赖关系。

- **创建环境**
- 创建特定名字的环境
  `conda create -n env_name`
- 使用特定版本的Python创建环境
  `conda create -n env_name python=3.4`
- 使用特定包创建环境
  `conda create -n env_name pandas`
- 用 environment.yml 配置文件创建环境
  `conda env create -f nvironment.yml`
  environment.yml 文件：
  `name: stats2 channels: - javascript dependencies: - python=3.4 # or 2.7 - bokeh=0.9.2 - numpy=1.9.* - nodejs=0.10.* - flask - pip: - Flask-Testing`

- **导出环境文件**`environment`

- 导出`environment.yml`环境文件

- - 激活需要导出文件的环境

```
conda activate env_name
```

- - 导出

```
conda env_name export > environment.yml
```

- **激活环境**

```
conda activate env_name
```

- **停用环境**

```
conda deactivate env_name
```

- **查看环境**（当前环境用*表示）

```
conda info -envs
```

- **删除环境**

```
conda remove --n env_name
```