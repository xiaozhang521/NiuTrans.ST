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
 */

#include "Generator.h"
#include <iostream>

using namespace nts;
using namespace std;

namespace s2t
{
    /* constructor */
    Generator::Generator()
    {
        config = NULL;
        model = NULL;
        seacher = NULL;
        outputBuf = new XList;
    }

    /* de-constructor */
    Generator::~Generator()
    {
        if (config->inference.beamSize > 1)
            delete (BeamSearch*)seacher;
        else
            delete (GreedySearch*)seacher;
        delete outputBuf;
    }

    /* initialize the model */
    void Generator::Init(S2TConfig& myConfig, S2TModel& myModel)
    {
        cout << "----- Generator Init -----" << endl;
        model = &myModel;
        config = &myConfig;

        cout << "--- Generator Init End ---" << endl;
    }

    bool Generator::TestTranslate()
    {
        // Pad audio 30s at right

        // extract fbank feature

        // Load test data from file for test
        XTensor test_audio_pad;
        InitTensor2D(&test_audio_pad, 80, 4100, X_FLOAT);
        FILE* audioFile = fopen(config->inference.inputFN, "rb");
        if (audioFile) {
            test_audio_pad.BinaryRead(audioFile);
        }
        test_audio_pad.Dump(stderr, NULL, 10);
        
        if (audioFile)
            fclose(audioFile);

        // preprocess

        


        return 1;
    }

}