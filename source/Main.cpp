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
 * $Modify for test by Erfeng HE 2023-09
 */

#include <iostream>
#include <fstream>

#include "./nmt/Config.h"
#include "./nmt/train/Trainer.h"
#include "./nmt/translate/Translator.h"

#include "./s2t/WaveLoader.h"
#include "./s2t/FeatureWindow.h"

using namespace nmt;
using namespace s2t;

int main(int argc, const char** argv)
{
    //----------------------------load wave----------------------------
    ifstream inFile("C:\\Code\\VS\\NiuTrans.ST\\test.wav", ios::in | ios::binary);
    if (!inFile) {
        cout << "error no file" << endl;
        return 0;
    }
    class WaveInfo wave;
    class WaveData data;
    data.Read(inFile);
    struct FrameExtractionOptions opt;
    class OfflineFeatureTpl<FrameExtractionOptions> oft(opt);
    XTensor out;
    oft.ComputeFeatures(data.Data(), data.SampFreq(), 1.0, &out);
   

    // ----------------------------load wave--------------------------------
    //std::ios_base::sync_with_stdio(false);
    //std::cin.tie(NULL);

    //if (argc == 0)
    //    return 1;
    ///* load configurations */
    //S2TConfig config(argc, argv);
    //S2TModel model;
    //model.InitModel(config);
    //config.showConfig();
    //Generator generator;
    //generator.Init(config, model);
    //generator.generate(); 
    // 
    /*****************************Old entrance******************************/
    //srand(config.common.seed);

        Trainer trainer;
        trainer.Init(config, model);
        trainer.Run();
    }

    /* translation */
    else if (strcmp(config.translation.inputFN, "") != 0) {

        /* disable gradient flow */
        DISABLE_GRAD;

        NMTModel model;
        model.InitModel(config);

        Translator translator;
        translator.Init(config, model);
        translator.Translate();
    }

    else {
        fprintf(stderr, "Thanks for using NiuTrans.NMT! This is an effcient\n");
        fprintf(stderr, "neural machine translation system. \n\n");
        fprintf(stderr, "   Run this program with \"-train\" for training!\n");
        fprintf(stderr, "Or run this program with \"-input\" for translation!\n");
    }

    return 0;
}