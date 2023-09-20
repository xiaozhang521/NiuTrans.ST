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

/* 
 * the test sample (a pair of utterance and text which contains the results of ASR and NMT).
 * the result of NMT(tranlSeq) could be NULL.
*/

	struct Sample
	{

		/* the index of the pair */
		int index;

		/* the key of buckets */
		int bucketKey;

		/* the sequence of audio (a list of tokens) */
		IntList* audioSeq;

		/* the sequence of target sentence (a list of tokens) */
		IntList* tgtSeq;

		/* the sequence of translated sentence (a list of tokens) */
		IntList* translSeq;

		/* constructor */
		Sample(IntList* a, IntList* tg, IntList* tr = NULL, int myKey = -1);

		/* de-constructor */
		~Sample();
	};

	/* the base class of datasets used in Niutrans.ST*/
	class AudioDataSetBase : public DataDistributeBase
	{
	public:
		/* frame counter, decided on duration */
		int frameCounter;

		/* the number of target sequences */
		int tgtCounter;

		/* the number of translated sequences*/
		int translCounter;

		/* the buffer of sequences */
		XList* seqBuf;

		/* the buffer of audio */
		XList* audioBuf;

		/*the configuration of ST system*/
		NMTConfig* config; //We will change it to STConfig in furture

	public:
		/* get the maximum audio sentence length in a range of buffer */
		int MaxAudioLen(int begin, int end);

		/* get the maximum target sentence length in a range of buffer */
		int MaxTgtLen(int begin, int end);

		/* get the maximum translated sentence length in a range of buffer */
		int MaxTranslLen(int begin, int end);

		/* sort the input by audio sentence length (in ascending order) */
		void SortByAudioLengthAscending();

		/* sort the input by target sentence length (in ascending order) */
		void SortByTgtLengthAscending();

		/* sort the input by translated sentence length (in ascending order) */
		void SortByTranslLengthAscending();

		/* sort the input by audio sentence length (in descending order) */
		void SortByAudioLengthDescending();

		/* sort the input by target sentence length (in descending order) */
		void SortByTgtLengthDescending();

		/* sort the input by translated sentence length (in descending order) */
		void SortByTranslLengthDescending();

		/* release the samples in a buffer */
		void ClearBuf();

	public:
		/* constructor */
		AudioDataSetBase();

		/* load the samples into the buffer (a list) */
		virtual
			bool LoadBatchToBuf() = 0;

		/* initialization function */
		virtual
			void Init(NMTConfig& myConfig, bool isTraining) = 0;

		/* load a sample from the file stream  */
		virtual
			Sample* LoadSample() = 0;

		/* load a mini-batch from the buffer */
		virtual
			bool GetBatchSimple(XList* inputs, XList* golds = NULL) = 0;

		/* de-constructor */
		~AudioDataSetBase();
	};

} /* end of nmt namespace */
#endif /* __AUDIO_DATASET_H__ */