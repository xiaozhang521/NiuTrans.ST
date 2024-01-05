 # **NiuTrans.ST**
This project aim to build a toolkit for speech-to-text task (e.g. Speech Recognition and Speech Translation). It is powered by the [NiuTensor](https://github.com/NiuTrans/NiuTensor#linux%E5%92%8Cmacos). 
## What's NEW
We support the multilingual speech Recognition now! Try it now (https://asr-demo.niutrans.com)!
## Features
- Light and fast, all the code is developed by the C++ and CUDA.
- Independent development. well-optimized for both CPUs and GPUs.

# Install


#### *Files -----------------------------------------------------------------------------------*

####    *|---- data (raw audio & its list, saved audio feature & its list)*

####    *|---- model (converted model file & vocab)*

####    *|---- NiuTrans.ST (model source written in C++)*

####    *|---- output* 

####    *|---- pic*

####    *|---- script (run.sh & pre/post-process script)*

####    *|---- test (test files)*

#### *--------------------------------------------------------------------------------------------*



### You need to prepare C++(for NiuTrans.ST) & Python(for pre/post-process) environments for this project first.

<br />

## Compile NiuTrans.ST project

    recommand environment
    - GCC=8.X.X (>=5.4.0 should be ok)
    - CUDAToolkit=11.3 with cuDNN (11.3 or 11.7 is work, but does not support version 12)
    - Cmake=3.2X.X


### 1. enter the project directory

    cd /PROJECT_DIR/NiuTrans.ST
    mkdir build & cd build


### 2. Execute the CMake command
    cmake ../ -DUSE_CUDA=ON -DUSE_CUDNN=ON -DGPU_ARCH=P -DCUDA_TOOLKIT_ROOT='D:/CUDA Toolkit/v11.7'

You can use `nvidia-smi -q | grep "Architecture"` to search your 'DGPU_ARCH'. More details please refer to [NiuTensor](https://github.com/NiuTrans/NiuTensor#linux%E5%92%8Cmacos)


### 3. Compile project
    make -j$(nproc)


### 4. Test
we provide test files in `/PROJECT_DIR/test`. 

    /PROJECT_DIR/bin/NiuTrans.ST -config /path/to/test/config.txt


# Optional toolkits
    toolkit (unifying the audio format)
    - ffmpeg 
    python package (Using python to process the audio if you prefer)
    - torch (https://pytorch.org/get-started/previous-versions/) 
    - numpy
    - webrtcvad
    - librosa
    - soundfile


# Run


### **Modify the paths in following files**
    /PROJECT_DIR/script/run.sh
    *If you get some bugs that cannot be fixed, consider using an absolute path.*


### **Run `run.sh` in [Git Bash](https://www.cnblogs.com/woods1815/p/11026658.html)**
    cd /PROJECT_DIR/script/
    bash run.sh


**To use your own audio files, you can modify the audio path in `/PROJECT_DIR/data/test-clean-head.tsv`.**


# Details about Run.sh

    ${python} ${FeatureExactor} -l ${audio_tsv} -o ${bin_tsv} -p ${bin_dir} -v ${vad_mode} -d 30.0
FeatureExactor with exact feature from raw audio files that were listed in audio_tsv. Features will be saved in bin_tsv. Use vad_mode to choose the strength of VAD, '0' means not using VAD.

    ${NiuTrans} -config ${config}
NiuTrans.ST

    ${python} ${TokenDecoder} -v ${vocab} -t ${token} > ${output_sentence}
TokenDecoder will decode the tokens outputed from the model to words.


# Details about the config in Run.sh

    config_string="-fbank 80        // fixed in 80
               -dev 0               // device id
               -beam 2              // size of beam seach(use greedy search when 1)
               -model ${model}      // path to converted model file
               -tgtvocab ${vocab}   // path to converted vocab file
               -maxlen 224          
               -lenalpha 1.0
               -sbatch 2            // batch_size(sentence)
               -wbatch 300000       // max_tokens
               -lang en             // language, only support en and zh(with fixed prompt)
               -input ${input}      // path to the list of input feature
               -output ${output}"   // path to save output tokens

Here, you can use '**-input**' to enter **features** of multiple audio data, or use '**-inputAudio**' to input a single **raw** audio data like:

    skip the FeatureExactor step

    config_string="-fbank 80        // fixed in 80
               -dev 0               // device id
               -beam 2              // size of beam seach(use greedy search when 1)
               -model ${model}      // path to converted model file
               -tgtvocab ${vocab}   // path to converted vocab file
               -maxlen 224          
               -lenalpha 1.0
               -sbatch 2            // batch_size(sentence)
               -wbatch 300000       // max_tokens
               -lang en             // language, only support en and zh(with fixed prompt)
               -inputAudio /PROJECT_DIR/data/flac/672-122797-0000.flac     
                                    // path to a raw audio file
               -output ${output}"   // path to save output tokens

# Team members
Current team members are *Yuhao Zhang, Xiangnan Ma, Kaiqi Kou, Erfeng He, Chenghao Gao*.

We'd like thank *Prof. XIAO, Tong* and *Prof. ZHU, Jingbo* for their support.


###  <br /><br />
# Other resources
### NiuTrans OpenSource(https://opensource.niutrans.com/home/index.html)(https://github.com/NiuTrans)
### NiuTensor (https://github.com/NiuTrans/NiuTensor)
### NiuTrans.NMT (https://github.com/NiuTrans/NiuTrans.NMT)
### MT Book (https://github.com/NiuTrans/MTBook)