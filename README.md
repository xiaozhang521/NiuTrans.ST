## 部署
`cuda:`11.7,
`cudnn:` 8.9.5,
`gcc:`8.5
```sh
# 编译环境变量
# gcc                                                                                   
export PATH=/home/gaochenghao/tools/gcc-8.5.0/bin:$PATH                                 
export LD_LIBRARY_PATH=/home/gaochenghao/tools/gcc-8.5.0/lib64/:$LD_LIBRARY_PATH        
#export CC=/home/gaochenghao/tools/gcc-8.5.0/bin/gcc
#export CXX=/home/gaochenghao/tools/gcc-8.5.0/bin/g++ 

# cmake
export PATH=/home/gaochenghao/tools/cmake-3.25.0-linux-x86_64/bin/:$PATH   

# cuda                                                                                  
export PATH=/home/gaochenghao/tools/cuda-11.7/bin:$PATH                                 
export LD_LIBRARY_PATH=/home/gaochenghao/tools/cuda-11.7/lib64/:$LD_LIBRARY_PATH

```

```sh
# 前端处理依赖
pytorch
pip install webrtcvad,librosa

```

## 编译

```sh
# 在CMakeLists.txt中配置 CUDA_TOOLKIT_ROOT路径 (也可以直接通过编译命令引入)

# 编译命令
mkdir build
cd build
cmake .. -DUSE_CUDA=ON -DUSE_CUDNN=ON \
-DCMAKE_C_COMPILER="/home/gaochenghao/tools/gcc-8.5.0/bin/gcc" \
-DCMAKE_CXX_COMPILER="/home/gaochenghao/tools/gcc-8.5.0/bin/g++" \
-DCUDA_TOOLKIT_ROOT="/home/gaochenghao/tools/cuda-11.7"
make -j 32
```
## 运行

``` sh
# config_file里面使用绝对路径，防止出现加载失败
./NiuTrans.NMT -config /home/gaochenghao/data/NiuTransData/test_config_file.txt
```

## 速度测试

- `测试音频`: /data/zhangyuhao/librispeech/LibriSpeech/test-clean/672/122797/672-122797-0000.flac
- `时常`: 4s
- `音频内容`: Out in the woods to the nice little fir tree.
- `前处理`: pad到30s的
- `测试脚本 (Whisper)` : sample/WhisperSpeedTest.py
- `测试脚本 (Our)` : sample/run_SpeedTest.sh

**循环10次耗时 (ms)**

| 模型      | tiny | base | large |
|---------|------|------|-------|
| Whisper | 850  |      |       |
| Our     | 1893 |      |       |


**循环20次耗时 (ms)**

| 模型      | tiny | base | large |
|---------|------|------|-------|
| Whisper | 1573 |      |       |
| Our     | 2456 |      |       |



## 报错

```sh
# 解码时出现如下错误
[INFO] loaded 1 sentences
----- S2TBeamSearch Search -----
----- Encoding -----
Could not load library libcublasLt.so.11. Error: libcublasLt.so.11: cannot open shared object file: No such file or directory

# 解决方法：添加CUDA的环境变量

```