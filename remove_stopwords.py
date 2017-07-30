import os


class removal(object):
    def __init__(self, infile, outfile):
        self.infile = infile
        self.outfile = outfile

    def remove(self):
        stopwords = open("chigga_stopwords.txt", 'r').read().splitlines()
        out_file = open(self.outfile, 'w')
        for line in open(self.infile, "r"):
            words = line.split(" ")
            out_line = ""
            for word in words:
                if word not in stopwords:
                    out_line += word + " "
            out_file.writelines(out_line)
        out_file.close()

    def test(self):
        for line in open(self.infile, "r"):
            print(line)
            print(line.split(' '))

if __name__ == '__main__':
    removal("train_neg.txt1", "train_neg.txt2").remove()
    removal("test_neg.txt1", "test_neg.txt2").remove()
    removal("train_pos.txt1", "train_pos.txt2").remove()
    removal("train_unsup.txt1", "train_unsup.txt2").remove()
    # removal("test_pos.txt1", "test_pos.txt2").remove()



