#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
'''
@File    :   blender_trans.py
@Time    :   2025/07/24 11:12:41
@Author  :   StarCheng
@Version :   1.0
@Site    :   https://star-cheng.github.io/Blog/
'''
# sudo apt-get install -y blender
# blender --python mujoco/tools/blender_trans.py
import bpy
import os

stl_dir = r"./description/tiangong_description/meshes_back"
save_dir = r"./description/tiangong_description/meshes"
stl_listdir = os.listdir(stl_dir)
for stl_name in stl_listdir:
    # 导入 STL 文件
    input_file = os.path.join(stl_dir, stl_name)
    output_file = os.path.join(save_dir, stl_name)
    # 清除场景中的所有对象（可选）
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    # 导入 STL 文件
    bpy.ops.import_mesh.stl(filepath=input_file)

    # 获取导入的对象（假设为场景中的唯一对象）
    obj = bpy.context.selected_objects[0]  

    # 设置对象为活动对象
    bpy.context.view_layer.objects.active = obj

    # 以当前活动对象为例，获取面数
    if obj and obj.type == 'MESH':
        face_count = len(obj.data.polygons)
        print(f"对象 '{obj.name}' 的面数为：{face_count}")


    # 确保对象是Mesh类型
    if obj.type == 'MESH':
        # 添加减面修改器
        modifier = obj.modifiers.new(name='Decimate', type='DECIMATE')
        # if face_count < 10000:
        #     modifier.ratio = 1.0
        # else:
        #     modifier.ratio = 10000 / face_count
        modifier.ratio = 0.1

        # 选择对象，确保它被选中
        obj.select_set(True)

        # 导出 STL 文件
        bpy.ops.export_mesh.stl(filepath=output_file)
    else:
        print("导入的对象不是Mesh类型")