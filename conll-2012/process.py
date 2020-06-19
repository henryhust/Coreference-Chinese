with open("百度百科字嵌入", encoding="utf8") as fr, open("char.vocab", "w", encoding="utf8") as fw:
    set0 = set()
    for i, line in enumerate(fr):
        line = line.rstrip().split(" ")
        if line[0] != "":
            fw.write(str(line[0])+"\n")
            if str(line[0]) in set0:
                print(line[0])
            else:
                set0.add(line[0])
    print(len(set0))
print("finished")
