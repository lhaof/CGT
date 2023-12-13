#!/usr/bin env python3
# coding=utf-8

import numpy as np
from math import floor, ceil
import torch

def interpolate(first_value: float, second_value: float, ratio: float) -> float:
    """Interpolate with a linear weighted sum."""
    return first_value * (1 - ratio) + second_value * ratio


def get_array_value(x: int, y: int, array: np.ndarray):
    """Returns the value of the array at position x,y."""
    return array[y, x]


def bilinear_interpolation(x: float, y: float, img: np.ndarray) -> float:
    """Returns the bilinear interpolation of a pixel in the image.

    :param x: x-position to interpolate
    :param y: y-position to interpolate
    :param img: image, where the pixel should be interpolated
    :returns: value of the interpolated pixel
    """
    if x < 0 or y < 0:
        raise ValueError("x and y pixel position have to be positive!")
    if img.shape[1] - 1 < x:
        x = img.shape[1] - 1 
    
    if img.shape[0] - 1 < y:
        y = img.shape[1] - 1 
        

    x_rounded_up = int(ceil(x))
    x_rounded_down = int(floor(x))
    y_rounded_up = int(ceil(y))
    y_rounded_down = int(floor(y))

    ratio_x = x - x_rounded_down
    ratio_y = y - y_rounded_down

    interpolate_x1 = interpolate(
        get_array_value(x_rounded_down, y_rounded_down, img),
        get_array_value(x_rounded_up, y_rounded_down, img),
        ratio_x,
    )
    interpolate_x2 = interpolate(
        get_array_value(x_rounded_down, y_rounded_up, img),
        get_array_value(x_rounded_up, y_rounded_up, img),
        ratio_x,
    )
    interpolate_y = interpolate(interpolate_x1, interpolate_x2, ratio_y)

    return interpolate_y
def bilinear_mask(image,single_centers,countour_points,scale=0):
    image = image.permute(1,2,0)
    length = single_centers.shape[0]
    countour_points = countour_points.reshape(countour_points.shape[0],countour_points.shape[1])
    numbers = len(countour_points) // len(single_centers)
    out_features = []
    countour_points = countour_points * scale
    for j in range(length):
        concat_feat = []

        for k in range(numbers*j,numbers*(j+1)):
            if countour_points[k][0] > 0 and countour_points[k][1] > 0:
                countour_point = bilinear_interpolation(countour_points[k][1],countour_points[k][0],image)
            else:
                countour_point = torch.zeros((64)).cuda()
            concat_feat.append(countour_point)
        
        
        #print(torch.cat(concat_feat).shape)
        out_features.append(torch.cat(concat_feat))
    out_features = torch.stack(out_features)
    #out_centers = torch.stack(out_centers)
    #print(out_centers.shape)
    return out_features
    
def bilinear_centers(image,single_centers,scale=0):
    
    image = image.permute(1,2,0)
    length = single_centers.shape[0]
    #print(length)
    #print(image.shape)
    out_centers = []
    single_centers = single_centers * scale
    for j in range(length):
        #print(single_centers[j])
        out = bilinear_interpolation(single_centers[j][1],single_centers[j][0],image)
        out_centers.append(out)
    out_centers = torch.stack(out_centers)
    return out_centers    
    
def bilinear_edge(image,single_centers,scale=0):
    
    image = image.permute(1,2,0)
    length = single_centers.shape[0]
    #print(length)
    #print(image.shape)
    out_centers = []
    single_centers = single_centers * scale
    for j in range(length):
        #print(single_centers[j])
        if single_centers[j][0] > 0 and single_centers[j][1] > 0:
            out = bilinear_interpolation(single_centers[j][1],single_centers[j][0],image)
            #print(out.shape)
        else:
            out = torch.zeros((64)).cuda()
        out_centers.append(out)
    #print(len(out_centers))
    out_centers = torch.stack(out_centers,0)
    return out_centers
