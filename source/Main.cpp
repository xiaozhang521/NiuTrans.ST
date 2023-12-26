/* NiuTrans.NMT - an open-source neural machine translation system.
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
 * $Created by: Chi Hu (huchinlp@gmail.com) 2021-11-06
 */

#include <iostream>

#include "./s2t/S2TConfig.h"
#include "./nmt/train/Trainer.h"
#include "./s2t/S2TModel.h"
#include "./s2t/generate/Generator.h"



#define CLOCKS_PER_SEC ((clock_t)1000)

#include <ctime>
#include "./utils/timer.h"

using namespace nmt;
using namespace s2t;
using namespace nts;

int main(int argc, const char **argv) {
    if (argc == 0)
        return 1;

    DISABLE_GRAD;
    /* load configurations */
    S2TConfig *config = new S2TConfig(argc, argv);
    S2TModel *model = new S2TModel();

    model->InitModel(config);


    Generator generator;
    //std::cout << (strlen(config.inference.inputFN) == 0) << (strcmp(config.extractor.inputAudio, "") == 0) << std::endl;
    CheckNTErrors(strcmp(config->inference.inputFN, "") || strcmp(config->extractor.inputAudio, ""),
                  "Giving input path to choose offline or input audio to choose online decoding");
    // Choosing online inference with speech extractor
    if (strlen(config->extractor.inputAudio) != 0) {
        //struct FbankOptions fOpts(config);
        //class FbankComputer computer(fOpts);
        //class OfflineFeatureTpl<FbankComputer> oft(computer);
        generator.Init(config, model, false);
    }
        // Choosing offline inference with batch decoding
    else if (strlen(config->inference.inputFN) != 0) {
        generator.Init(config, model);
    } else {
        CheckNTErrors((strlen(config->inference.inputFN) != 0 || strlen(config->extractor.inputAudio) != 0),
                      "Giving input path to choose offline or input audio to choose online decoding");
    }

    // generator.Generate();

    clock_t start, finish;
    double duration;
    start = clock();

    for (int i = 1; i <= 10; ++i) {
        char language[MAX_NAME_LEN];
        char file[MAX_PATH_LEN];
        strcpy(language, "zh");
        string path="/home/gaochenghao/data/NiuTransData/data/zh/"+ to_string(i)+".wav";
        // 使用 c_str() 将 std::string 转换为 C 风格字符串
        strncpy(file, path.c_str(), MAX_PATH_LEN - 1);
        // 确保字符串以 null 字符结尾
        file[MAX_PATH_LEN - 1] = '\0';

        generator.Interact(language,file , FALSE);

    }
    cudaThreadSynchronize();
    finish = clock();
    duration = (double) (finish - start) / CLOCKS_PER_SEC;
    printf("Time:\t%.2fs\n", duration / 1000.0);


    printf("=============time_consume=============\n");
    printf("Conv+GELU:\t%.2fs\n", time_conv1d / 1000.0);
    printf("LayerNorm:\t%.2fs\n", time_ln / 1000.0);
    printf("FFN:\t\t%.2fs\n", time_ffn / 1000.0);
    printf("Attention:\t%.2fs\n", time_attn / 1000.0);
    printf("Attention(Mul):\t%.2fs\n", time_attn_mul / 1000.0);
    printf("Output:\t\t%.2fs\n", time_output / 1000.0);

    printf("=============time_consume=============\n");
    //generator.generate(); 
    // 
    /*****************************Old entrance******************************/
    //srand(config.common.seed);

    ///* training */
    //if (strcmp(config.training.trainFN, "") != 0) {

    //    NMTModel model;
    //    model.InitModel(config);

    //    Trainer trainer;
    //    trainer.Init(config, model);
    //    trainer.Run();
    //}

    ///* translation */
    //else if (strcmp(config.translation.inputFN, "") != 0) {

    //    /* disable gradient flow */
    //    DISABLE_GRAD;

    //    NMTModel model;
    //    model.InitModel(config);

    //    Translator translator;
    //    translator.Init(config, model);
    //    translator.Translate();
    //}
    //else {
    //    fprintf(stderr, "Thanks for using NiuTrans.NMT! This is an effcient\n");
    //    fprintf(stderr, "neural machine translation system. \n\n");
    //    fprintf(stderr, "   Run this program with \"-train\" for training!\n");
    //    fprintf(stderr, "Or run this program with \"-input\" for translation!\n");
    //}

    return 0;
}