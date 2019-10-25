import os
import random
import argparse

def split(scp,out_path,max_len = 10000):
    with open(scp,'r') as f:
        newfile = []
        lines = f.readlines()
        random.shuffle(lines)
        for j in range(0,len(lines),max_len):
            newfile.append(lines[j:j+max_len])
    outfile = []
    for i in range(len(newfile)):
        outname = out_path+"/"+scp+"_"+str(i)+".scp"
        with open(outname,'w') as f:
            newfile[i].sort()
            f.writelines(newfile[i])
        outfile.append(outname)
    return outfile   

def genconfig(name,type,outfile,label,aux_label):
    with open(name,'w') as f:
        f.write("clean_source:\n")
        for i in range(len(outfile)):
            f.write("  %d:\n"%(i+1))
            f.write("    type: %s\n"%(type))
            f.write("    wav: %s\n"%(outfile[i]))
            f.write("    label: %s\n"%(label))
            f.write("    aux_label: %s\n"%(aux_label))

# label = "/data2/tmpt/chenggaofeng/librispeech_s5_2018_11_28_for_pytorch_x/train-960-pdf-ids-tri6b.txt"
# aux_label = "/data2/tmpt/chenggaofeng/librispeech_s5_2018_11_28_for_pytorch_x/train-960-trans-ids-tri6b.txt"
# out = split("new_train.scp","/data2/tmpt/chenggaofeng/pykaldi2-master-20190924/example/my_librispeech/data/split")
# genconfig("data.yaml","Librispeech",out,label,aux_label)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-scp_file",default='', type=str, help="path of scp to split")
    parser.add_argument("-exp_dir")
    parser.add_argument("-data_type", default='Librispeech', type=str)
    parser.add_argument("-split_length", default=10000, type=int)
    parser.add_argument("-label", default='', type=str, help="path of label files")
    parser.add_argument("-aux_label", default='', type=str, help="path of aux_label files")
    args = parser.parse_args()

    if not os.path.isdir(args.exp_dir):
        os.makedirs(args.exp_dir)

    out = split(args.scp_file,args.exp_dir,args.split_length)
    genconfig(args.exp_dir+"/data.yaml",args.data_type,out,args.label,args.aux_label)


if __name__ == "__main__":
    main()