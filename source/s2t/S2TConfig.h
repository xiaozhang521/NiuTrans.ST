/* NiuTrans.S2T - an open-source speech-to-text system.
 * Copyright (C) 2020 NiuTrans Research. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */


 /*
  * $Created by: Yuhao Zhang (yoohao.zhang@gmail.com) 2023-09-19
  *
  */

#ifndef __CONFIG_S2T__
#define __CONFIG_S2T__

//#include <chrono>
#include <vector>
#include <string>
#include "../niutensor/tensor/XConfig.h"
#include "../nmt/Config.h"

using namespace std;
using namespace nts;
using namespace nmt;
#ifndef INT_TO_int
#define INT_TO_int
#define INT16 int16_t
#define INT32 int32_t
#define INT64 int64_t
#define UINT16 int16_t
#define UINT32 uint32_t
#define UINT64 uint64_t
#define UINT uint32_t
#define TRUE true
#define FALSE false
#endif

/* the s2t namespace */
namespace s2t
{

    #define MAX_NAME_LEN 20

    union LanguageUnion {
        int languageToken;
        char language[MAX_NAME_LEN];
    };

    /* model configuration */
    class S2TModelConfig : public XConfig
    {
    public:
        /* the dimension of fbank */
        int fbank;
        /*Sub-sampling for speech feature*/
        /* TODO!!!  this following vector configs, now they are static for whisper.*/
        int nConv = 2;
        vector<int> convKernel = { 3,3 };
        vector<int> convStride = { 1,2 };
    public:
        /* load configuration from the command */
        void Load(int argsNum, const char** args);
    };

    /* whisper decoding configuration */
    class WhisperDecConig : XConfig
    {
        /*TODO*/
    public:

        char task[MAX_NAME_LEN];
        LanguageUnion language;     // zh
        float temperature;
        float noSpeechThreshold;
        float logProbThreshold;
        float compRatioThreshold;
        bool withoutTimeStamps;

    public:
        /* load configuration from the command */
        void Load(int argsNum, const char** args);

        void InitLanguageToken();

    };

    /* inference configuration */
    class InferenceConfig : public TranslationConfig
    {
    public:

    };

    /* Feature extraction config*/
    class ExtractionConfig : public XConfig {
    public:

        bool useEnergy;
        float energyFloor;
        bool rawEnergy;
        bool htkCompat;
        bool useLogFbank;
        bool usePower;
        bool oneSide;
        char inputAudio[MAX_PATH_LEN];

        float sampFreq;
        float frameShiftMs;
        float frameLengthMs; 
        float chunkLengthMs;
        float dither;  
        float preemphCoeff;
        bool removeDcOffset;
        char windowType[MAX_NAME_LEN]; 
        bool roundToPowerOfTwo;
        float blackmanCoeff;
        bool snipEdges;
        bool allowDownsample;
        bool allowUpsample;
        int maxFeatureVectors;
        int torchPaddingLength; 
        char padMod[MAX_NAME_LEN];

        INT32 numBins;
        float lowFreq;
        float highFreq;
        float vtlnLow; 
        float vtlnHigh; 
        bool debugMel;
        bool htkMode;
        char customFilter[MAX_PATH_LEN];

    public:
        void Load(int argsNum, const char** args);

    };

    /* configuration of the s2t project  */
    class S2TConfig : public NMTConfig
    {
    public:

        /* Feature extraction config*/
        ExtractionConfig extractor;

        /* model configuration */
        S2TModelConfig s2tmodel;

        /* common configuration */
        // CommonConfig common;

        /* training configuration */
        // TrainingConfig training;

        /* inference configuration */
        InferenceConfig inference;
        /* whisper decoding configuration */
        WhisperDecConig whisperdec;

    public:
        /* load configuration from the command */
        S2TConfig(int argc, const char** argv);
        /* load configuration from a file */
        int LoadFromFile(const char* configFN, char** args);
        /* print configurationle */
        void showConfig();
    };
} /* end of the s2t namespace */

#endif /* __CONFIG_S2T__ */
