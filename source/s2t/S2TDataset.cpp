/* NiuTrans.S2T - an open-source speech to text system.
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
  * $Created by: Yuhao Zhang (yoohao.zhang@gmail.com) 2023-10-13
  */


#include <algorithm>
#include "S2TDataSet.h"

using namespace nts;

/* the s2t namespace */
namespace s2t {

/* get the maximum source sentence length in a range of buffer */
int AudioDataSetBase::MaxAudioLen(int begin, int end) {
    CheckNTErrors((end > begin) && (begin >= 0)
        && (end <= buf->count), "Invalid range");
    int maxLen = 0;
    for (int i = begin; i < end; i++) {
        TensorList* audioSeq = ((TripleSample*)buf->Get(i))->audioSeq;
        maxLen = MAX(int(audioSeq->Size()), maxLen);
    }
    return maxLen;
}

/* get the maximum target sentence length in a range of buffer */
int AudioDataSetBase::MaxTgtLen(int begin, int end) {
    CheckNTErrors((end > begin) && (begin >= 0)
        && (end <= buf->count), "Invalid range");
    int maxLen = 0;
    for (int i = begin; i < end; i++) {
        IntList* tgtSent = ((TripleSample*)buf->Get(i))->tgtSeq;
        maxLen = MAX(int(tgtSent->Size()), maxLen);
    }
    return maxLen;
}

/* get the maximum source sentence length in a range of buffer */
int AudioDataSetBase::MaxSrcLen(int begin, int end) {
    CheckNTErrors((end > begin) && (begin >= 0)
        && (end <= buf->count), "Invalid range");
    int maxLen = 0;
    for (int i = begin; i < end; i++) {
        IntList* srcSent = ((TripleSample*)buf->Get(i))->srcSeq;
        maxLen = MAX(int(srcSent->Size()), maxLen);
    }
    return maxLen;
}

/* sort the buffer by audio length (in ascending order) */
void AudioDataSetBase::SortByAudioLengthAscending() {
    stable_sort(buf->items, buf->items + buf->count,
        [](void* a, void* b) {
            return ((TripleSample*)(a))->audioSeq->Size() <
                ((TripleSample*)(b))->audioSeq->Size();
        });
}

/* sort the buffer by target sentence length (in ascending order) */
void AudioDataSetBase::SortByTgtLengthAscending()
{
    stable_sort(buf->items, buf->items + buf->count,
        [](void* a, void* b) {
            return ((TripleSample*)(a))->tgtSeq->Size() <
                ((TripleSample*)(b))->tgtSeq->Size();
        });
}

/* sort the buffer by source sentence length (in ascending order) */
void AudioDataSetBase::SortBySrcLengthAscending() {
    stable_sort(buf->items, buf->items + buf->count,
        [](void* a, void* b) {
            return ((TripleSample*)(a))->srcSeq->Size() <
                ((TripleSample*)(b))->srcSeq->Size();
        });
}

/* sort the buffer by audio length (in descending order) */
void AudioDataSetBase::SortByAudioLengthDescending() {
    stable_sort(buf->items, buf->items + buf->count,
        [](void* a, void* b) {
            return ((TripleSample*)(a))->audioSeq->Size() >
                ((TripleSample*)(b))->audioSeq->Size();
        });
}

/* sort the buffer by target sentence length (in descending order) */
void AudioDataSetBase::SortByTgtLengthDescending()
{
    stable_sort(buf->items, buf->items + buf->count,
        [](void* a, void* b) {
            return ((TripleSample*)(a))->tgtSeq->Size() >
                ((TripleSample*)(b))->tgtSeq->Size();
        });
}

/* sort the buffer by source sentence length (in descending order) */
void AudioDataSetBase::SortBySrcLengthDescending() {
    stable_sort(buf->items, buf->items + buf->count,
        [](void* a, void* b) {
            return ((TripleSample*)(a))->srcSeq->Size() >
                ((TripleSample*)(b))->srcSeq->Size();
        });
}

/*
clear the buffer
>> buf - the buffer (list) of samples
*/
void AudioDataSetBase::ClearBuf()
{
    bufIdx = 0;
    for (int i = 0; i < buf->count; i++) {
        TripleSample* sample = (TripleSample*)buf->Get(i);
        delete sample;
    }
    buf->Clear();
}

/* constructor */
AudioDataSetBase::AudioDataSetBase()
{
    fc = 0;
    wc = 0;
    sc = 0;
    bufIdx = 0;
    config = NULL;
    buf = new XList();
}

/* de-constructor */
AudioDataSetBase::~AudioDataSetBase()
{
    if (buf != NULL) {
        ClearBuf();
        delete buf;
    }
}

/* constructor */
TripleSample::TripleSample(TensorList* a, IntList* s, IntList* t, int myKey) {
    index = -1;
    audioSeq = a;
    srcSeq = s;
    tgtSeq = t;
    bucketKey = myKey;
}

/* de-constructor */
TripleSample::~TripleSample() {
    if (audioSeq != NULL)
        delete audioSeq;
    if (srcSeq != NULL)
        delete srcSeq;
    if (tgtSeq != NULL)
        delete tgtSeq;
}

}