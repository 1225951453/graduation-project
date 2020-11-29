import os

#data_by_myself
data_pth = r"F:/Nonsegmented/"
def Process():
    idx = 0
    Dir = os.listdir(data_pth)
    with open("F:/plant/plant_data.txt", 'w', encoding='utf-8') as f:
        for v in Dir:
            for var in os.listdir(os.path.join(data_pth,v)):
                pth = os.path.join(data_pth,v,var)
                # l1 = var.split("_")[0]
                # l2 = l1.split("_")[0]
                pth = pth + ',' + str(idx)
                f.write(pth + '\n')
            idx = idx + 1

Process()
# data_pth = r"F:/cifar-10-batches-py/train_image_cifar10/"
# def Process():
#     d1 = os.listdir(data_pth)
#     num = 0
#     with open("C:/Users/WANGYONG/Desktop/internet+/classification_code/cifar10.txt",'w',encoding='utf-8') as f1:
#         for i in d1:
#             if i.split('.')[1] == 'txt':
#                 with open(data_pth + i,'r') as f2:
#                     line = f2.readline()
#                     f2.close()
#                 f1.write(data_pth + i.split(".")[0]+ '.jpg' +',' + str(line.split(" ")[0]) + '\n')

# Process()



# import os
# data_pth = r"F:/cifar-10-batches-py/test_image_cifar10/"
# def Process():
#     d1 = os.listdir(data_pth)
#     num = 0
#     with open("C:/Users/WANGYONG/Desktop/internet+/classification_code/test_cifar10.txt",'w',encoding='utf-8') as f1:
#         for i in d1:
#             if i.split('.')[1] == 'txt':
#                 with open(data_pth + i,'r') as f2:
#                     line = f2.readline()
#                     f2.close()
#                 f1.write(data_pth + i.split(".")[0]+ '.jpg' +',' + str(line.split(" ")[0]) + '\n')
#
# Process()