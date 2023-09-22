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

/* the s2t namespace */
namespace s2t
{

    #define MAX_NAME_LEN 20

    /* model configuration */
    class S2TModelConfig : public ModelConfig
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
        /* print configurationle */
        void showConfig();
    };

    /* inference configuration */
    class InferenceConfig : public TranslationConfig
    {
    public:
        /* print configurationle */
        void showConfig();
    };


    /* configuration of the s2t project  */
    class S2TConfig : public NMTConfig
    {
    public:
        /* model configuration */
        S2TModelConfig model;

        /* common configuration */
        // CommonConfig common;

        /* training configuration */
        // TrainingConfig training;

        /* inference configuration */
        InferenceConfig inference;

    public:
        /* load configuration from the command */
        S2TConfig(int argc, const char** argv);
        /* load configuration from a file */
        int LoadFromFile(const char* configFN, char** args);
        /* print configurationle */
        void showConfig();
    };
} /* end of the s2r namespace */

#endif /* __CONFIG_S2T__ */