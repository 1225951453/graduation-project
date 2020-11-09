import os

data_pth = "C:/Users/WANGYONG/Desktop/internet+/data/lvluo_baidu/"
def Process():
    Dir = os.listdir(data_pth)
    with open("C:/Users/WANGYONG/Desktop/internet+/classification_code/data.txt", 'w', encoding='utf-8') as f:
        for v in Dir:
            for var in os.listdir(os.path.join(data_pth,v)):
                if var.split('.')[1] == 'jpg':

                    pth = os.path.join(data_pth,v,var)
                    l1 = var.split("_")[1]
                    # l2 = l1.split("_")[0]
                    pth = pth + ',' + l1
                    f.write(pth + '\n')

Process()