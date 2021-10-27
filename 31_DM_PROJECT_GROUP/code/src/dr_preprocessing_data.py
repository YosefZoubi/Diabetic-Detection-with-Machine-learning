import os
from os import listdir
from os.path import isfile, join
from PIL import Image
Image.LOAD_TRUNCATED_IMAGES = True
from skimage import io
from skimage.transform import resize, rotate
import numpy as np
import pandas as pd
import time
import cv2


local_path = '/content/drive/MyDrive/eye_dr_cnn/'

train_path = local_path + 'data/train/'
train_labels_csv = local_path + 'labels/trainLabels_new.csv'

output_img_size = 224
output_path = local_path + 'data/train-resized-224/'
output_labels_master = local_path + 'labels/trainLabels_new_master.csv'
output_labels_master_v2 = local_path + 'labels/trainLabels_new_master_224_v2.csv'
output_train_data_npy = local_path = 'data/X_train_224.npy'


def crop_and_resize_images(path, new_path, cropx, cropy, img_size=224):
    if not os.path.exists(new_path):
        os.makedirs(new_path)

    onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
    total = 0
    for item in onlyfiles:
        img = io.imread(path+item)
        y,x,channel = img.shape
        startx = x//2-(cropx//2)
        starty = y//2-(cropy//2)
        img = img[starty:starty+cropy,startx:startx+cropx]
        img = resize(img, (img_size,img_size))
        io.imsave(str(new_path + item), img)
        total += 1
        print("\t-Saving: ", item, total)


def find_black_images(file_path, df):
    lst_imgs = [l for l in df['image']]
    return [1 if np.mean(np.array(Image.open(file_path + img))) == 0 else 0 for img in lst_imgs]


def rotate_images(file_path, degrees_of_rotation, lst_imgs):
    for l in lst_imgs:
        img = io.imread(file_path + str(l) + '.jpeg')
        img = rotate(img, degrees_of_rotation)
        io.imsave(file_path + str(l) + '_' + str(degrees_of_rotation) + '.jpeg', img)    


def mirror_images(file_path, mirror_direction, lst_imgs):
    for l in lst_imgs:
        img = cv2.imread(file_path + str(l) + '.jpeg')
        img = cv2.flip(img, 1)
        cv2.imwrite(file_path + str(l) + '_mir' + '.jpeg', img)


def convert_images_to_arrays_train(file_path, df):
    lst_imgs = [l for l in df['train_image_name']]
    return np.array([np.array(Image.open(file_path + img)) for img in lst_imgs])


def step1_crop_and_resize_images():
    crop_and_resize_images(path=train_path, new_path=output_path, cropx=1800, cropy=1800, img_size=output_img_size)


def step2_drop_black_images():
    trainLabels = pd.read_csv(train_labels_csv)
    trainLabels['image'] = [i + '.jpeg' for i in trainLabels['image']]
    trainLabels['black'] = np.nan
    trainLabels['black'] = find_black_images(output_path, trainLabels)
    trainLabels = trainLabels.loc[trainLabels['black'] == 0]
    trainLabels.to_csv(output_labels_master, index=False, header=True)


def step3_rotate_images():
    trainLabels = pd.read_csv(output_labels_master)
    dest_folder = output_path

    trainLabels['image'] = trainLabels['image'].str.rstrip('.jpeg')
    trainLabels_no_DR = trainLabels[trainLabels['level'] == 0]
    trainLabels_DR = trainLabels[trainLabels['level'] >= 1]

    lst_imgs_no_DR = [i for i in trainLabels_no_DR['image']]
    lst_imgs_DR = [i for i in trainLabels_DR['image']]

    # Mirror Images with no DR one time
    print("\t-Mirroring Non-DR Images")
    mirror_images(dest_folder, 1, lst_imgs_no_DR)

    # Rotate all images that have any level of DR
    print("\t-Rotating 90 Degrees")
    rotate_images(dest_folder, 90, lst_imgs_DR)

    print("\t-Rotating 120 Degrees")
    rotate_images(dest_folder, 120, lst_imgs_DR)

    print("\t-Rotating 180 Degrees")
    rotate_images(dest_folder, 180, lst_imgs_DR)

    print("\t-Rotating 270 Degrees")
    rotate_images(dest_folder, 270, lst_imgs_DR)

    print("\t-Mirroring DR Images")
    mirror_images(dest_folder, 0, lst_imgs_DR)
 

def step4_reconcile_labels():
    trainLabels = pd.read_csv(output_labels_master)
    lst_imgs = [f for f in listdir(output_path) if isfile(join(output_path, f))]

    new_trainLabels = pd.DataFrame({'image': lst_imgs})
    new_trainLabels['image2'] = new_trainLabels.image

    # Remove the suffix from the image names.
    new_trainLabels['image2'] = new_trainLabels.loc[:, 'image2'].apply(lambda x: '_'.join(x.split('_')[0:2]))

    # Strip and add .jpeg back into file name
    new_trainLabels['image2'] = new_trainLabels.loc[:, 'image2'].apply(
        lambda x: '_'.join(x.split('_')[0:2]).strip('.jpeg') + '.jpeg')

    new_trainLabels.columns = ['train_image_name', 'image']

    trainLabels = pd.merge(trainLabels, new_trainLabels, how='outer', on='image')
    trainLabels.drop(['black'], axis=1, inplace=True)
    trainLabels = trainLabels.dropna()

    print("\t-Writing CSV")
    trainLabels.to_csv(output_labels_master_v2, index=False, header=True)


def step5_image_to_array():
    labels = pd.read_csv(output_labels_master_v2)

    print("\t-Writing Train Array")
    X_train = convert_images_to_arrays_train(output_path, labels)

    print("\t-Saving Train Array")
    np.save(output_train_data_npy, X_train)


def run_step(name, step_func):
    print(name)
    start_time = time.time()
    step_func()
    print("Completed")
    print("--- %s seconds ---" % (time.time() - start_time))


if __name__ == '__main__':
    start_time = time.time()
    print('\n===================================\n')
    run_step('crop_and_resize_images', step1_crop_and_resize_images)
    print('\n===================================\n')
    run_step('drop_black_images', step2_drop_black_images)
    print('\n===================================\n')
    run_step('rotate_images', step3_rotate_images)
    print('\n===================================\n')
    run_step('reconcile_labels', step4_reconcile_labels)
    print('\n===================================\n')
    run_step('image_to_array', step5_image_to_array)
    print('\n===================================\n')
    print("Completed")
    print("--- %s seconds ---" % (time.time() - start_time))
    