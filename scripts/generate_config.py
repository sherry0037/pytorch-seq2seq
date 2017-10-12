from itertools import product
import random
import argparse
import os


train = "data/nucle/train/data_p.txt"
dev = "data/nucle/dev/data_p.txt"
expt = "./experiment/nucle/"

learning_rate = [0.1, 0.01, 0.001]
hidden_size = [256, 512, 1024]
n_layers = [1, 2]
batch_size = [32, 64, 128]
teacher_forcing_rate = [0, 0.2, 0.4]
drop_out = [0.2, 0.4]
weight_decay = [0, 0.2, 0.4]
embedding_size = [200, 400]
parameters = [learning_rate, hidden_size, n_layers, batch_size, teacher_forcing_rate, 
                drop_out, weight_decay, embedding_size]
parameters_names = ["learning_rate", "hidden_size", "n_layers", "batch_size", "teacher_forcing_rate", 
                    "drop_out", "weight_decay", "embedding_size"]

all_combination = [p for p in product(*parameters)]

def get_config(k=50, combination=all_combination, output_file="./config.ini"):
    combination = random.sample(combination, k)
    with open(output_file, "w") as f:
        for i in range(k):
            f.write("[%d]\n"%i)
            f.write("%s = %s\n"%("train", train))
            f.write("%s = %s\n"%("dev", dev))
            f.write("%s = %s\n"%("expt", expt+"model%d/"%i))
            for pi in range(len(parameters_names)):
                f.write("%s = %s\n"%(parameters_names[pi], combination[i][pi]))
            f.write("\n")
    print ("Finish writing file %s."%output_file)

def write_bash(k=50, time="0-6", mem=30, config="config.ini", preprocess=False, output_dir="./bash_scripts/"):
    try:
        os.makedirs(output_dir)
    except OSError as e:
        pass
    for model in range(k):
        output_file = output_dir+"model%d"%model
        with open(output_file, "w") as f:
            f.write("#!/bin/bash\n\n#SBATCH --nodes=1\n\n#SBATCH --ntasks-per-node=1\n\n#SBATCH --gres=gpu:2\n\n")
            f.write("#SBATCH --time=%s\n\n"%time)
            f.write("#SBATCH --mem=%dGB\n\n"%mem)
            if int(time[0]) == 0 and int(time[-1]) <= 6:
                f.write("#SBATCH --qos=express\n\n")
            f.write("module load cuda\n\nmodule load cudnn/v6\n\nmodule load pytorch/0.2.0p3-py27\n\nmodule load python\n\n")

            f.write("MODEL=%d\n"%model)
            f.write("CONFIG=%s\n\n"%config)

            f.write("SECONDS=0\n\n")
            f.write("python examples/train.py --model $MODEL --config $CONFIG\n")
            p = "-p" if preprocess else " "
            f.write("python examples/decode.py %s --model $MODEL --config $CONFIG\n"%p)
            #f.write("python examples/evaluate.py --model $MODEL --config $CONFIG\n\n")
            f.write("diff=$SECONDS\n")
            f.write("echo \"$(($diff / 3600))h $((($diff / 60) % 60))m $(($diff % 60))s elapsed.\"")
    print ("Finish writing %d files at %s."%(k, output_dir))

def main(args):
    random.seed(args.seed)
    get_config(k=args.k, output_file=args.output_dir+args.output_name)
    write_bash(k=args.k, time=args.time,config=args.output_name, preprocess=True, mem=args.mem)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generage configeration using random search.")
    parser.add_argument("-k", default=100, dest="k", type=int, help="Number of models. Default: 100")
    parser.add_argument("-m", default=100, dest="mem", type=int, help="Memory to allocate. Default: 100")
    parser.add_argument("-t", default="0-6", dest="time",  help="Memory to allocate. Default: 0-6")
    parser.add_argument("--random_seed", dest="seed", default=128, type=int)
    parser.add_argument("--output_dir", "-d", dest="output_dir", default="./examples/", 
                         help="Output dirctory for the configurations. Default: ./examples/")
    parser.add_argument("--config", "-n", dest="output_name", default="config.ini",
                         help="Name of the config file. Default: config.ini")
    args = parser.parse_args()
    main(args)






