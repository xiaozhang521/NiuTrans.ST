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
#include "./nmt/translate/Translator.h"
#include "./s2t/S2TModel.h"
#include "./s2t/generate/Generator.h"
#include "./s2t/S2TVocab.h"
#include "niutensor/tensor/function/GELU.h"

#include <ctime>
#include "./utils/timer.h"

using namespace nmt;
using namespace s2t;
using namespace nts;

int main(int argc, const char** argv)
{
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(NULL);

    if (argc == 0)
        return 1;
    /* load configurations */
    S2TConfig config(argc, argv);
    S2TModel model;
    model.InitModel(config);
    //config.showConfig();

    //model.TestDumpParams(&model.encoder->selfAtts[1].weightQ);

    //cout << "Tgt Vocab File: " << config.common.tgtVocabFN << endl;
    //S2TVocab vocab;
    //vocab.Load(config.common.tgtVocabFN);
    // vocab.ShowVocab();
    //vocab.Test();


    Generator generator;
    generator.Init(config, model);
    generator.Generate();

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