import os
import csv


basepath = 'vehicle/images'

imgList = [[], [], []]          # [train, val, test]
labelList = [[], [], []]
contextList = [[], [], []]

extensions = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp', '.gif')
select_dict = {'airplane':['in_sunrise', 'with_pilot'], 
               'bicycle':['in_garage', 'velodrome'], 
               'boat':['cross_bridge', 'at_wharf'], 
               'bus':['at_yard', 'aside_tree'], 
               'car':['at_park', 'on_track'], 
               'helicopter':['on_sea', 'in_forest'], 
               'motorcycle':['in_garage', 'on_track'], 
               'train':['subway', 'in_forest'], 
               'truck':['in_race', 'in_forest']}

for entry in os.listdir(basepath):

    classPath = os.path.join(basepath, entry)
    for context in os.listdir(classPath):

        itemPath = os.path.join(classPath, context)
        for item in os.listdir(itemPath):
            if item.lower().endswith(extensions):
                if context == select_dict[entry][0]:
                    imgList[1].append(os.path.join(entry, context, item))
                    labelList[1].append(entry)
                    contextList[1].append(context)
                elif context == select_dict[entry][1]:
                    imgList[2].append(os.path.join(entry, context, item))
                    labelList[2].append(entry)
                    contextList[2].append(context)
                else:
                    imgList[0].append(os.path.join(entry, context, item))
                    labelList[0].append(entry)
                    contextList[0].append(context)

train_rows = zip(imgList[0], labelList[0], contextList[0])
val_rows = zip(imgList[1], labelList[1], contextList[1])
test_rows = zip(imgList[2], labelList[2], contextList[2])

with open('vehicle/vehicle_train_ood.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerows(train_rows)
with open('vehicle/vehicle_val_ood.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerows(val_rows)
with open('vehicle/vehicle_test_ood.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerows(test_rows)











