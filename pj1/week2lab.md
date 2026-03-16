# 本地部署deepseek-R1

1. [下载安装ollama](https://ollama.com/download)

2. 检测是否安装完成
   ```cmd
   ollama -v
   ```

3. 拉取deepseek-R1 1.5b，或者其他版本[deepseek-r1](https://ollama.com/library/deepseek-r1:1.5b)

   ```cmd
   ollama run deepseek-r1:1.5b
   ```

4. 安装[Page Assit](https://chromewebstore.google.com/detail/page-assist-%E6%9C%AC%E5%9C%B0-ai-%E6%A8%A1%E5%9E%8B%E7%9A%84-web/jfgfiigpkhlkbnfnbobbkinehhfdhndo)浏览器插件（chrome、edge）

# 开发环境配置

1. 安装[vscode](https://code.visualstudio.com/)或[cursor](https://www.cursor.com/)

   > cursor是基于vscode开发的ai编辑器，在vscode的基础上增加了原生的ai功能，需要自行搭配api-key使用

2. 安装[miniconda](https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe)并创建虚拟环境

   1. conda 使用国内镜像

       ```cmd
       conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
       conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free
       conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
       conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/pro
       conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2
       conda config --set show_channel_urls yes
       pip config set global.index-url https://mirrors.ustc.edu.cn/pypi/simple/
       
       # switch to default
       conda config --remove-key channels
       pip config unset global.index-url
       ```

   2. [conda 常用指令](https://zhuanlan.zhihu.com/p/202698464)

       ```cmd
       # 创建环境
       conda create --name your_env_name python=3.10
       
       # 安装包
       pip install $name1(=$version1) ($name2=$version2...)
       #或者
       conda install $name1(=$version1) ($name2=$version2...)
       # 卸载包
       conda uninstall xxx
       ```

   3. 操作步骤

       1. 创建虚拟环境
       2. 使用国内镜像
       3. 安装torch、jupyter

3. 安装vscode插件：Python、Jupyter

4. 安装cuda、HIP等SDK

# 可选配置

+ 使用typora处理markdown文件
+ 使用wsl进行本地linux环境开发 [教程](https://zhuanlan.zhihu.com/p/438255467)
  + 将[wsl-ubuntu22.04lts](https://apps.microsoft.com/detail/9pn20msr04dw)等微软商店链接复制到[微软商店解析](https://store.rg-adguard.net/)，可以加速下载
