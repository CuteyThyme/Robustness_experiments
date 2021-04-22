# coding=utf-8

# ACE -> CoNLL format
def get_CONLL_format(ori_path, new_path):
    with open(ori_path, "r") as read_f:
        with open(new_path, "w") as write_f:
            for line in read_f:
                newline = ""
                if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                    newline = line
                else:
                    splits = line.split(" ")
                    newline = splits[0] + " O " + "O " + splits[-1]
                write_f.writelines(newline)


# Ontonotes -> CoNLL format
def generate_collection(ori_path, new_path):
    text = ""
    with open(ori_path, 'r') as read_f:
        flag = None
        for line in read_f.readlines():
            l = line.strip()
            l = ' '.join(l.split())
            ls = l.split(" ")
            if len(ls) >= 11:
                word = ls[3]
                pos = ls[4]
                cons = ls[5]
                ori_ner = ls[10]
                ner = ori_ner
                # print(word, pos, cons, ner)
                if ori_ner == "*":
                    if flag==None:
                        ner = "O"
                    else:
                        ner = "I-" + flag
                elif ori_ner == "*)":
                    ner = "I-" + flag
                    flag = None
                elif ori_ner.startswith("(") and ori_ner.endswith("*") and len(ori_ner)>2:
                    flag = ori_ner[1:-1]
                    ner = "B-" + flag
                elif ori_ner.startswith("(") and ori_ner.endswith(")") and len(ori_ner)>2 and flag == None:
                    ner = "B-" + ori_ner[1:-1]

                text += " ".join([word, 'O', 'O', ner]) + '\n'
                # print(text)
            else:
                text += '\n'
        text += '\n'
        # break

    with open(new_path, 'w') as write_f:
        write_f.write(text)
