from model.model import CoopNets
from opts import opts
import os




os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def main():
    opt=opts().parse()
    model=CoopNets(opt)
    swich = 7
    if swich == 0:
        model.test()
    elif swich == 1:
        model.impute()
    else:
        model.train()

if __name__=='__main__':
    main()

