## 部署

`cuda:`11.7,
`cudnn:` 8.9.5,
`gcc:`8.5

`注意`:gcc5.4在编译FeatureWindow相关cpp时会报错

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
./NiuTrans.ST -config /home/gaochenghao/data/NiuTransData/test_config_file.txt

# mp3 to wav
ffmpeg -i 10.mp3 -f wav -ac 1 -ar 16000 10.wav
```

## 速度测试

- `测试脚本 (Whisper)` : sample/WhisperSpeedTest.py
- `测试脚本 (Our)` : sample/run_SpeedTest.sh
- `测试音频`: /data/zhangyuhao/librispeech/LibriSpeech/test-clean/672/122797/672-122797-0000.flac
- `时常`: 4s
- `音频内容`: Out in the woods to the nice little fir tree.
- `前处理`: pad到30s的
- `batch`: 1, `beam`: 1

**循环1次耗时 (ms)**

| 模型               | tiny | large |
|------------------|------|-------|
| Whisper          | 222  | 667   |
| Our              | 1340 | 1935  |
| Whisper (beam=2) | 235  | 764   |
| Our (beam=2)     | 1345 | 1924  |

**循环10次耗时 (ms)**

| 模型               | tiny | large |
|------------------|------|-------|
| Whisper          | 870  | 5703  |
| Our              | 1688 | 7391  |
| Whisper (beam=2) | 977  | 6718  |
| Our (beam=2)     | 1647 | 7583  |

**循环20次耗时 (ms)**

| 模型               | tiny | large |
|------------------|------|-------|
| Whisper          | 1588 | 11317 |
| Our              | 2085 | 13522 |
| Whisper (beam=2) | 1795 | 13136 |
| Our (beam=2)     | 1994 | 13932 |

----
- `测试音频`: long_bbc.mp3
- `时常`: 30s
- `音频内容`: Live from London, this is BBC News. Britain's former Prime Minister Boris Johnson apologises for the pain and loss suffered in the UK during the coronavirus pandemic. Can I just say how glad I am to be at this inquiry and how sorry I am for the pain and the loss and the suffering. Heavy fighting is taking place across Gaza as Israeli tanks close in on three areas of
- `前处理`: pad到30s的
- `batch`: 1, `beam`: 1
- `GPU`: 3090

**循环10次耗时 (ms)**

| 模型               | tiny | large |
|------------------|------|-------|
| Whisper          | -    | 31389 |
| Our              | -    | 25819 |
| Whisper (beam=2) | -    | 34225 |
| Our (beam=2)     | -    | 31716 |

## 报错

```sh
# 解码时出现如下错误
[INFO] loaded 1 sentences
----- S2TBeamSearch Search -----
----- Encoding -----
Could not load library libcublasLt.so.11. Error: libcublasLt.so.11: cannot open shared object file: No such file or directory

# 解决方法：添加CUDA的环境变量

```