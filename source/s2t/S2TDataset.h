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
 * We define the class datasets for audio here.
 * every sub-datasets(e.g.,train, valid, etc.) of dataset should have a manifest
 * It will be overrided for inferrence, just for now.
 * 
 * $Created by: HE Erfeng (heerfeng1023@gmail.com) 2023-09
 */

#ifndef __AUDIO_DATASET_H__
#define __AUDIO_DATASET_H__

/* 
 * We may need a new config.h and a new namespace for ST
*/

#include "../nmt/Config.h"
#include "../niutensor/train/XBaseTemplate.h"

using namespace std;

/* the nmt namespace */
namespace nmt 
{

/* the training or test sample (a pair of utterance and text) */
	struct Sample
	{

		/* the index of the pair */
		int index;

		/* the key of buckets */
		int bucketKey;

		/* the sequence of audio (a list of tokens) */
		IntList* audioSeq;

		/* the sequence of text (a list of tokens) */
		IntList* txtSeq;

	};

	/* the base class of datasets used in Niutrans.ST*/
	class AudioDataSetBase : public DataDistributeBase
	{
	public:

	};
}
#endif /* __AUDIO_DATASET_H__ */