# KVC2

# Build
运行以下命令编译kvc2，注意可能需要 sudo 权限安装一些依赖
```shell
git clone https://github.com/kvcache-ai/kvc2
cd kvc2
./install_deps.sh
mkdir build
cd build
cmake ..
make -j && make install
```
编译完成后会生成`build/output`，包含`kvc2_ext.cpython-312-x86_64-linux-gnu.so`和`kvc2_utils.py`方便调用。

<!-- # Test
运行以下命令测试kvc2，需要指定一个 disk path 作为测试目录。
```shell
./unit_test.sh ${DISK_PATH}
```
或者运行 python 的测试文件
```shell
python test/pytest_mem_read.py 
``` -->

# Troubleshooting
在 Python 环境运行时，可以需要在 conda 中安装相关的依赖。
```shell
conda install -c conda-forge gcc_linux-64 gxx_linux-64
```

也可以尝试设置一下环境变量，然后再运行。
```shell
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libffi.so.7 
```


