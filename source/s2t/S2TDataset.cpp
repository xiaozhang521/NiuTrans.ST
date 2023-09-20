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
  * $Created by: HU Chi (huchinlp@gmail.com) 2021-06
  */


#include <algorithm>
#include "DataSet.h"

using namespace nts;

namespace nmt {
    /* constructor */
    Sample::Sample(IntList* a, IntList* tg, IntList* tr = NULL, int myKey = -1)
    {
        index = -1;
        audioSeq = a;
        tgtSeq = tg;
        translSeq = tr;
        bucketKey = myKey;
    }

    /* de-constructor */
    Sample::~Sample()
    {
        if (srcSeq != NULL)
            delete srcSeq;
        if (tgtSeq != NULL)
            delete tgtSeq;
    }
}