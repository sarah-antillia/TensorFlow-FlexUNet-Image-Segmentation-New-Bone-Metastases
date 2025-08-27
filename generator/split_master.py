# Copyright 2025 antillia.com Toshiyuki Arai
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# split_master.py
# 2025/08/27


import os
import sys
import glob
import shutil
import cv2
import traceback
import random

def split_master(images_dir, masks_dir, output_dir):
  image_files = glob.glob(images_dir + "/*.jpg")
  random.seed(137)
  random.shuffle(image_files)
  num = len(image_files)
  num_train = int(num * 0.8)
  num_valid = int(num * 0.15)
  num_test  = int(num * 0.05)
  print("num_train {}".format(num_train))
  print("num_valid {}".format(num_valid))
  print("num_test  {}".format(num_test ))

  train_files = image_files[:num_train]
  valid_files = image_files[num_train:num_train+ num_valid]
  test_files  = image_files[num_train+ num_valid:]
  train_dir   = os.path.join(output_dir, "train")
  valid_dir   = os.path.join(output_dir, "valid")
  test_dir    = os.path.join(output_dir, "test")
  copy(train_files, masks_dir, train_dir)
  copy(valid_files, masks_dir, valid_dir)
  copy(test_files,  masks_dir, test_dir )


def copy(image_files, masks_dir, dataset_dir):
  out_images_dir = os.path.join(dataset_dir, "images")
  out_masks_dir  = os.path.join(dataset_dir, "masks")

  if not os.path.exists(out_images_dir):
    os.makedirs(out_images_dir)
  if not os.path.exists(out_masks_dir):
    os.makedirs(out_masks_dir)

  for image_file in image_files:
    basename = os.path.basename(image_file)
    png_name = basename.replace(".jpg", ".png")

    image = cv2.imread(image_file)
    out_image_filepath = os.path.join(out_images_dir, png_name)
    cv2.imwrite(out_image_filepath, image)

    print("Copied {} to {}".format(image_file, out_images_dir))

    mask_file = image_file.replace("/images/", "/masks")
    mask = cv2.imread(mask_file)

    out_mask_filepath = os.path.join(out_masks_dir, png_name)
  
    cv2.imwrite(out_mask_filepath, mask)
    print("Copied {} to {}".format(mask_file, out_masks_dir))


if __name__ == "__main__":
  try:
    images_dir = "./new_dataset/image/"
    masks_dir  = "./new_dataset/lesion mask/"
    output_dir = "../dataset/NBM/"

    if os.path.exists(output_dir):
      shutil.rmtree(output_dir)
    if not os.path.exists(output_dir):
      os.makedirs(output_dir)

    split_master(images_dir, masks_dir, output_dir)

  except:
    traceback.print_exc()
