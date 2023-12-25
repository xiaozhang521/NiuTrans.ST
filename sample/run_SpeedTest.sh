# This bash contains the progress of "Vad&feature exacting(python)", "NiuTrans encoder+decoder(C++)" and "Token decode(python)"
export PATH=/home/gaochenghao/tools/cuda-11.7/bin:$PATH
export LD_LIBRARY_PATH=/home/gaochenghao/tools/cuda-11.7/lib64/:$LD_LIBRARY_PATH

# script path
ROOT=/home/gaochenghao/NiuTrans.ST
# audio,model path
DATA=/home/gaochenghao/data/NiuTransData

FeatureExactor=${ROOT}/sample/audio_extractor.py
NiuTrans=${ROOT}/bin/NiuTrans.NMT
TokenDecoder=${ROOT}/sample/covertTokens.py


#audio_wav=${DATA}/data/zh_12345_more.wav
#audio_wav=/home/gaochenghao/workplace/data/NiuTransData/data/672-122797-0000.flac
audio_wav=/home/gaochenghao/data/NiuTransData/data/long_bbc.mp3

bin_tsv=${DATA}/data/test-vad-wav_more.tsv

bin_dir=${DATA}/data/bin
vad_mode=1

config=${DATA}/config.txt
input=${bin_tsv}
output=${DATA}/output/output.txt

vocab=${DATA}/model/vocab_whisper_utf
token=${output}

#model=${DATA}/model/whisper_tiny_niutrans_s2t.bin
model=${DATA}/model/whisper_large_v2_niutrans_s2t.bin

timestamp_s=$(date "+%s%N")

# Vad&feature exacting(python)
python3 ${FeatureExactor} -a ${audio_wav} -o ${bin_tsv} -p ${bin_dir} -v ${vad_mode} -d 30.0

# NiuTrans encoder+decoder(C++)

config_string="-fbank 80
               -bigatt false
               -dev 0
               -beam 1
               -model ${model}
               -tgtvocab ${DATA}/model/vocab_whisper_utf
               -maxlen 224
               -lenalpha 1.0
               -sbatch 1
               -wbatch 2000
               -lang en
               -input ${input}
               -output ${output}"

echo ${config_string} > ${config}
rm ${output}

${NiuTrans} -config ${config}

# Token decode(python)
python3 ${TokenDecoder} -v ${vocab} -t ${token}

timestamp_e=$(date "+%s%N")

duration=$(expr $(expr $timestamp_e - $timestamp_s) / 1000000)

echo "Total using time is ${duration} ms"