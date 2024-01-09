import os
import argparse

def main():

    parser = argparse.ArgumentParser(prog='Token Decoder', description='Decoder tokens to Words according to the vocab.')
    
    parser.add_argument("-v", "--vocab", help="Vocab file", type=str)
    parser.add_argument("-t", "--token", help="Token file", type=str)

    args = parser.parse_args()
    
    file = "../test/output.txt"

    with open(file, 'r') as file:
        
        file_contents = file.read()
    
    # print(file_contents)

    tokensBatch = [ sen.split(' ') for sen in file_contents.split('\n')]

    # print(tokensBatch)

    vocab_file = args.vocab
    vocab_i2t = {}
    with open("../model/vocab_whisper_utf.bin", 'r') as vocab_file:
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

    for index, res in enumerate(batch):
        print("Sentence", index)
        print(res)


if __name__ == "__main__":
    main()
