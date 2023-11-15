import os
import argparse
from jiwer import wer
def main():

    parser = argparse.ArgumentParser(prog='Token Decoder', description='Decoder tokens to Words according to the vocab.')
    
    parser.add_argument("-v", "--vocab", help="Vocab file", type=str)
    parser.add_argument("-t", "--token", help="Token file", type=str)

    args = parser.parse_args()
    
    file = args.token

    with open(file, 'r') as file:
        
        file_contents = file.read()
    
    # print(file_contents)

    tokensBatch = [ sen.split(' ') for sen in file_contents.split('\n')]

    # print(tokensBatch)

    vocab_file = args.vocab
    vocab_i2t = {}
    with open(vocab_file, 'r') as vocab_file:
        vocab_contents = vocab_file.read()

    for token in vocab_contents.split('\n'):
        if token:
            pair = token.split('\t')
            if len(pair) == 2:
                vocab_i2t[pair[1]] = pair[0]
        
    # print(vocab_i2t)

    batch = []
    for sen in tokensBatch:
        if sen:
            sen_utf8 = b''
            for id in sen:
                if id:
                    sen_utf8 += eval(vocab_i2t[id])
            
            if sen_utf8:
                sen_unicode = sen_utf8.decode('utf-8')
                # print(sen)
                # print(sen_utf8)
                # print(sen_unicode)
                batch.append(sen_unicode)

    py=[]
    for index, res in enumerate(batch):
        print(index, res)
        py.append(res.strip().lower()[:-1])


    # hy=[]
    # with open("/home/gaochenghao/data/NiuTransData/data/test-vad.tsv",'r') as f_h:
    #     lines=f_h.readlines()[1:]
    #     for line in lines:
    #         h_txt=line.strip().split("\t")[3]
    #         hy.append(h_txt.strip().lower())
    #
    # if len()
    # error = wer(hy, py)
    # print("-"*100)
    # print("label:")
    # print(hy)
    # print("-"*100)
    # print("predict:")
    # print(py)
    # print("wer:{}".format(error))


if __name__ == "__main__":
    main()
