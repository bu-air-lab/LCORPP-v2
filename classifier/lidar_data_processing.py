# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 12:54:16 2020

@author: cckklt
"""

import numpy as np
import matplotlib.pyplot as plt
from plyfile import PlyData, PlyElement
from PIL import Image,ImageOps
import os

source_path_A="lidar_output_test/class_A/"
target_path_A="test_data/class_A/"
middle_path_A="preprocess_data_test/class_A/"
source_path_B="lidar_output_test/class_B/"
target_path_B="test_data/class_B/"
middle_path_B="preprocess_data_test/class_B/"
#project 3d object to 2d grayscale image


def image_project(file_name,source_path,target_path):     
      plydata = PlyData.read(source_path+file_name+'.ply')
      x,y=plydata.elements[0].data['x'],plydata.elements[0].data['y']
      fig=plt.figure()
      plt.scatter(x, y)
      fig.set_size_inches(28, 28, forward=True)
      #plt.imshow((x,y))
      plt.savefig(target_path+file_name+".jpg")
      #plt.show()


def image_crop(image_name,source_path,target_path):
      img=Image.open(source_path+image_name+".jpg")
      #print(img.format, img.size, img.mode)
      width,height=img.size

      #print(width,height)
      left=(2/5)*width
      upper=(1/10)*height
      right=(1/2)*width
      lower=(9/10)*height

      #print(left,upper,right,lower)
      img_crop=img.crop((left,upper,right,lower))
      #img_crop.show()
      img_processed=ImageOps.grayscale(img_crop)
      img_processed.save(target_path+image_name+".jpg")
      


#lidar_data_process("mesh","name_test")
def batch_process_project():
      data_files_class_A=os.listdir(source_path_A)
      data_files_class_B=os.listdir(source_path_B)

      count_A=len(data_files_class_A)
      count_B=len(data_files_class_B)
      print("There are",count_A,"class A instances")
      print("There are",count_B,"class B instances")
      
      for data_file in data_files_class_A:
            image_name=data_file.split(".")[0]
            image_project(image_name,source_path_A,middle_path_A)

      for data_file in data_files_class_B:
            image_name=data_file.split(".")[0]
            image_project(image_name,source_path_B,middle_path_B)

def batch_process_crop():
      middle_path_A_list=os.listdir(middle_path_A)
      middle_path_B_list=os.listdir(middle_path_B)
      for image in middle_path_A_list:
            image_name=image.split(".")[0]
            image_crop(image_name,middle_path_A,target_path_A)     
      for image in middle_path_B_list:
            image_name=image.split(".")[0]
            image_crop(image_name,middle_path_B,target_path_B)
batch_process_project()
batch_process_crop()
       
"""
plydata = PlyData.read("099895.ply")
x,y=plydata.elements[0].data['x'],plydata.elements[0].data['y']
fig=plt.figure()
fig.set_size_inches(28, 28, forward=True)
plt.scatter(x, y)
#plt.show()
plt.savefig("099895.jpg",quality=95, dpi=fig.dpi)
"""

"""
print(plydata.elements[0].data[0])
print(plydata.elements[0].name)
print(plydata.elements[0].data['x'])
print(plydata.elements[0].count)

"""