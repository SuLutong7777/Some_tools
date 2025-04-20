##############################################################
## 该文件将Spann3R得到的相机参数npy文件转换成Blender和LLFF格式 ##
##############################################################
import torch
import json
import os
import math
import shutil
import numpy as np
from pathlib import Path

### 读取Spann3R得到的npy文件
class Spann3r_Camera_data():
    def __init__(self):
        self.npy_path = "/home/sulutong/3R/spann3r-main/output/demo/llff/fern/fern.npy"
        self.score_pts = 1e-3
        # 字典格式
        self.all_data = self.load_npy_file()
        # 每张图片对应的名称
        self.name_order = self.extract_pose_order(self.all_data)
        print("name_order: ", self.name_order)
        # 相机外参 [N, 4, 4]
        self.all_c2w, self.cam_numb = self.extract_pose_c2w(self.all_data)
        # 相机内参 [N, 3, 3]
        self.all_intr = self.extract_intr_mat(self.all_data, self.cam_numb)
        # pca居中 pca_c2w:[N, 4, 4]  trans_mat:[4, 4]
        self.pca_c2w, self.trans_mat = self.transform_poses_pca(self.all_c2w)
        print("self.pca_c2w: ", self.pca_c2w.shape)
        # 点云提取 all_pts:[N, H*W, 3]  all_color:[N, H*W, 3]  mask_pts:[N, H*W, 1]
        self.all_pts, self.all_color, self.mask_pts = self.extract_point_clouds(self.all_data, self.trans_mat)
        # 图片尺寸
        self.img_h, self.img_w = self.all_intr[0][1, -1]*2, self.all_intr[0][0, -1]*2

    # 注意，数组是可以保存在npy文件中的，但是读出来不是字典
    # 需要转换才能变成字典，否则是0维数组(使用.item)
    def load_npy_file(self):
        data = np.load(self.npy_path, allow_pickle=True) # 允许使用 pickle 来加载对象。如果 .npy 文件中保存的是字典、列表、PyTorch 张量等复杂对象（而不仅仅是普通的数组），就需要设置为 True
        return data.item()

    # 提取位姿对应图片的名称
    def extract_pose_order(self, all_data):
        name_all = all_data["names"]
        return name_all
            
    # 提取相机的外参
    def extract_pose_c2w(self, all_data):
        pose_all = all_data["poses_all"]
        return pose_all, pose_all.shape[0]

    # 提取相机的内参
    def extract_intr_mat(self, all_data, cam_numb=None):
        intrinsic = all_data["intrinsic"]
        # 将内参矩阵复制cam_numb份
        if cam_numb is not None:
            intrinsic = np.tile(intrinsic[np.newaxis], [cam_numb, 1, 1])
        return intrinsic

    # 提取相机的点云信息
    def extract_point_clouds(self, all_data, trans_mat=None):
        conf_all = all_data["conf_all"]
        conf_sig_all = (conf_all-1) / conf_all
        mask = (conf_sig_all>self.score_pts).reshape(self.cam_numb, -1)
        # mask = all_data["masks_all"].reshape(self.cam_numb, -1) # DUSt3R
        # [N, H*W, 3]
        pts = all_data["pts_all"].reshape(self.cam_numb, -1, 3)
        # [N, H*W, 3]
        colors = all_data["images_all"].reshape(self.cam_numb, -1, 3)
        # 添加齐次坐标，变为 [B, H*W, 4]
        ones = np.ones((self.cam_numb, pts.shape[1], 1))
        pts_homo = np.concatenate([pts, ones], axis=-1)
        # 应用变换矩阵 
        if trans_mat is not None:
            pts_homo = np.einsum('ij,bnj->bni', trans_mat, pts_homo) # 批量张量操作
        # 返回转换后的 3D 坐标，去掉齐次坐标的第四列
        transformed_pts = pts_homo[..., :3]
        return transformed_pts, colors, mask

    # COLMAP产生的相机位姿的世界坐标系不一定是啥样
    # 这个操作将COLMAP生成的坐标系进行转换，变成以环绕中心为世界坐标系原点的全新分布坐标
    # pca是指主成分分析，Principal Component Analysis，一种数据降维方法
    # 主成分分析可以看这个：https://zhuanlan.zhihu.com/p/37777074
    # 这段代码实现PAC用的是上面这个链接中3.5的(1)方法
    # 输入poses为[N, 4, 4], 必须为c2w矩阵，不能为w2c
    def transform_poses_pca(self, poses_c2w):
        poses_c2w = torch.from_numpy(poses_c2w).to(torch.float32)
        # 获取所有相机的中心点
        trans = poses_c2w[:, :3, 3]
        # 取平均值
        trans_mean = torch.mean(trans, dim=0)
        # 中心化，相当于取所有点的平均中心为新坐标原点
        # 生成新的相机中心位置 [194, 3]
        trans = trans - trans_mean
        # 计算特征值eigval，和特征向量eigvec
        # 注意，这两个算出来是复数格式，有实部和虚部，即使虚部为0，也会保留
        # 所以这里要除去虚部(虚部全部算出来都是0)
        # trans.T @ trans: [3,3], 注意，这个过程在计算平移向量集合的协方差（正常有个除以n的系数，但是不影响特征向量）
        # eigval:[3], eigvec:[3,3]
        # eigval, eigvec = torch.linalg.eig(trans.T @ trans)
        # 转成Numpy做，pytorch版本的特征向量符号与Numpy不一致
        eigval, eigvec = np.linalg.eig(np.array(trans).T @ np.array(trans))
        eigval = torch.from_numpy(eigval)
        eigvec = torch.from_numpy(eigvec)
        # print(eigval, eigvec)
        # exit()
        # 对所有特征值进行从大到小的排序，获取排序的索引
        inds = torch.argsort(eigval.real, descending=True)
        # 同时排序特征向量
        # eigvec = eigvec[:, inds].real
        eigvec = eigvec[:, inds]
        # print(eigvec, "2222")
        # 将特征向量转置，构造投影矩阵，将所有坐标点投影到新的坐标系下
        # 这个新的坐标系的轴就是数据的主成分轴。
        # 这里eigvec为[3,3]，因为数据一共有三个主成分，分别为x,y,z，都需要保留，所以上面链接中的k值取3，就等同于不用筛选
        # eigvec中，每一列是特征向量，转置之后变成行，在进行投影的时候就是rot@trans，x,y,z维度能对应
        rot = eigvec.T
        # 保持坐标系变换后与原来规则相同
        # 在三维空间中，一个合法的旋转矩阵应该是正交的且行列式为1，这保证了坐标系变换保持了空间的右手规则。
        # 如果行列式小于0，表明旋转矩阵将导致坐标系翻转，违反了右手规则。
        # 一个矩阵的行列式（np.linalg.det(rot)）告诉我们这个矩阵是保持空间的定向（右手或左手）不变还是改变了空间的定向。具体来说：
        # 如果行列式大于0，说明变换后的坐标系保持原有的定向（即如果原坐标系是右手的，变换后仍然是右手的）。
        # 如果行列式小于0，说明变换后的坐标系改变了原有的定向（即从右手变为了左手，或从左手变为了右手）。
        if torch.linalg.det(rot) < 0:
            rot = torch.diag(torch.tensor([1.0, 1.0, -1.0])) @ rot

        # 构建完整的[R|T]变换矩阵，直接针对原始的pose信息，不再单纯考虑trans
        # 尺寸是[3, 4]
        transform_mat = torch.cat([rot, rot @ -trans_mean[:, None]], dim=-1)
        # 转为[4, 4]
        transform_mat = torch.cat([transform_mat, torch.tensor([[0, 0, 0, 1.]])], dim=0)
        # 整体RT矩阵转换[N, 4, 4]
        poses_recentered = transform_mat @ poses_c2w

        # 检查坐标轴方向
        # 检查在新坐标系中，相机指向的平均方向的y分量是否向下。如果是的话，这意味着变换后的位姿与常规的几何或物理约定（例如，通常期望的y轴向上）不符。
        if poses_recentered.mean(axis=0)[2, 1] < 0:
            poses_recentered = torch.diag(torch.tensor([1.0, -1.0, -1.0, 1.0])) @ poses_recentered
            transform_mat = torch.diag(torch.tensor([1.0, -1.0, -1.0, 1.0])) @ transform_mat
        # print(poses_recentered, "lll")
        # print(transform_mat, "eeee")

        # 对数据进行归一化，收敛到[-1, 1]之间
        scale_factor = 1. / torch.max(torch.abs(poses_recentered[:, :3, 3]))
        poses_recentered[:, :3, 3] *= scale_factor
        poses_recentered[:, 3, :] = torch.tensor([0.0, 0.0, 0.0, 1.0]).repeat(poses_recentered.shape[0], 1)
        transform_mat = torch.diag(torch.tensor([scale_factor] * 3 + [1])) @ transform_mat
        
        return poses_recentered, transform_mat

### 将Spann3R得到的npy文件转换为blender数据集格式
class Transform_spann3r_to_Blender():
    def __init__(self, mix_data):
        # 数据集原始根路径
        self.blender_root = "/home/A_DataSets2/nerf_llff_data" 
        self.mix_data = mix_data
        # 生成blender数据集保存根路径
        self.save_path = "/home/sulutong/3R/spann3r-main/zzz_data"
        self.data_name = "fern"
        # 数据集原始路径
        self.data_root = os.path.join(Path(self.blender_root), Path(self.data_name))
        # 获取训练数据
        self.train_dict = self.get_train_data()
        # 获取验证数据
        self.test_dict = self.get_test_data()
        self.save_param()
    
    def get_train_data(self):
        # 读取数据
        self.ori_w2c = self.mix_data.ori_w2c
        self.ori_c2w = self.mix_data.ori_c2w
        self.ori_intr = self.mix_data.ori_intr
        self.colors_all = self.mix_data.colors_gt
        self.train_names = self.mix_data.train_names
        # 整理数据
        # 相机的视场角
        camera_angle_x = self.generate_camera_angle_x(self.ori_intr)
        train_dict = {"camera_angle_x": camera_angle_x}
        # 整理帧数据
        frame_data = self.generate_frames_data(self.ori_w2c, self.train_names, mode="train")
        train_dict["frames"] = frame_data
        return train_dict
    
    def get_test_data(self):
        self.ori_intr = self.mix_data.ori_intr
        self.render_c2w = self.mix_data.ori_c2w
        self.render_w2c = self.inverse_RT(self.render_c2w)
        self.render_colors = self.mix_data.colors_gt
        self.test_names = self.mix_data.train_names
        # 整理数据
        # 相机的视场角
        camera_angle_x = self.generate_camera_angle_x(self.ori_intr)
        test_dict = {"camera_angle_x": camera_angle_x}
        # 整理帧数据
        frame_data = self.generate_frames_data(self.render_w2c, self.test_names, mode="test")
        test_dict["frames"] = frame_data        
        return test_dict

    def get_test_data_ori(self):
        self.ori_intr = self.mix_data.ori_intr
        self.render_c2w = self.mix_data.render_c2w
        self.render_w2c = self.inverse_RT(self.render_c2w)
        self.render_colors = self.mix_data.render_colors
        self.test_names = self.mix_data.test_names
        
        # 整理数据
        # 相机的视场角
        camera_angle_x = self.generate_camera_angle_x(self.ori_intr)
        test_dict = {"camera_angle_x": camera_angle_x}
        # 整理帧数据
        frame_data = self.generate_frames_data(self.render_w2c, self.test_names, mode="test")
        test_dict["frames"] = frame_data        

        return test_dict

    # 将相机内参保存成blender需要的格式
    def generate_camera_angle_x(self, train_intr):
        # 取一个
        operate_intr = train_intr[0]
        focal = operate_intr[0, 0]
        img_w_half = operate_intr[0, 2]
        tan_half_angle = img_w_half / focal
        half_angle = math.atan(tan_half_angle)
        camera_angle_x = 2*half_angle

        return camera_angle_x

    # 生成相机的帧数据
    def generate_frames_data(self, w2c_data, names, mode="train"):
        assert len(names) == len(w2c_data)
        data_list = []
        for cur_w2c, cur_name in zip(w2c_data, names):
            blender_w2c = self.blender_w2c_transform(cur_w2c.to("cpu"))
            blender_w2c_np = blender_w2c.tolist()
            cur_dict = {"transform_matrix":blender_w2c_np}
            # 处理路径
            split_name = cur_name.split(".")[0]
            cur_path = "./{}/{}".format(mode, split_name)
            cur_dict["file_path"] = cur_path
            data_list += [cur_dict]
        return data_list

    # blender格式的位姿转换函数
    def blender_w2c_transform(self, cv_pose):
        # cv_pose = new_pose
        pose_pad = cv_pose[3:, :].to(torch.float32)
        # cv_R = pose_R_new_inv
        cv_R = cv_pose[:3, :3].to(torch.float32)
        # cv_T = pose_T_new_inv
        cv_T = cv_pose[:3, 3:].to(torch.float32)
        pose_flip_R = torch.diag(torch.tensor([1.0,-1.0,-1.0]))
        # cv_R_new = pose_R_new
        cv_R_new = cv_R.T
        # blender_R = pose_R
        blender_R = cv_R_new @ pose_flip_R
        # blender_T = pose_T
        # pose_T = pose_T_new = pose_R_new @ -pose_T_new_inv
        blender_T = cv_R_new @ -cv_T
        new_pose = torch.cat([blender_R, blender_T], -1)
        new_pose = torch.cat([new_pose, pose_pad], dim=0)
        return new_pose

    # 旋转矩阵
    def inverse_RT(self, rt_mat):
        # print(c2w_mat.shape)
        ori_R = rt_mat[:, :3, :3]
        ori_t = rt_mat[:, :3, 3:]
        ori_R_T = ori_R.transpose(-1, -2)
        # print(ori_R_T.shape, ori_t.shape)
        T_new = -ori_R_T @ ori_t
        T_w2c = torch.eye(4, device='cuda')[None, ...].repeat(rt_mat.shape[0], 1, 1)
        T_w2c[:, :3, :3] = ori_R_T
        T_w2c[:, :3, 3:] = T_new

        return T_w2c

    # 根据生成的字典整理图片
    def copy_file(self, generate_dict, name):
        frames = generate_dict["frames"]
        folder_path = os.path.join(Path(self.save_path), Path(self.data_name), Path(name))
        # 创建文件夹
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)        
        # 逐帧复制图像
        for data in frames:
            img_name = data["file_path"].split("/")[-1]
            # 原始文件夹是train
            ori_img_path = os.path.join(Path(self.data_root), Path("images_4"), Path("{}.png".format(img_name)))
            tar_img_path = os.path.join(Path(folder_path), Path("{}.png".format(img_name)))
            shutil.copy(ori_img_path, tar_img_path)
        
        print('\nCopy Image Finish !!!!')

    def save_param(self):
        file_path = os.path.join(Path(self.save_path), Path(self.data_name))
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        filename_train = os.path.join(Path(file_path), Path("transforms_train.json"))
        filename_test = os.path.join(Path(file_path), Path("transforms_test.json"))

        with open(filename_train, 'w', encoding='utf-8') as f:
            json.dump(self.train_dict, f, ensure_ascii=False, indent=4)

        with open(filename_test, 'w', encoding='utf-8') as f:
            json.dump(self.test_dict, f, ensure_ascii=False, indent=4)

        # 注意，复制文件都是从train文件夹复制
        self.copy_file(self.train_dict, name="train")
        self.copy_file(self.test_dict, name="test")

        print('\nSave Param Finish !!!!')

### 将Spann3R得到的npy文件转换为llff数据集格式
class Transform_spann3r_to_LLFF():
    def __init__(self, mix_data):
        # 数据集原始路径
        self.llff_root = "/home/A_DataSets2/nerf_llff_data/"
        self.data_name = "fern"
        self.data_root = os.path.join(Path(self.llff_root), Path(self.data_name))
        # 生成新数据集路径
        self.save_path = "/home/sulutong/3R/spann3r-main/zzz_llff_data"
        self.mix_data = mix_data
        self.save_param()
    
    def save_param(self):
        file_path = os.path.join(Path(self.save_path), Path(self.data_name))
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        images_path = os.path.join(Path(file_path), Path("images_4"))
        if not os.path.exists(images_path):
            os.makedirs(images_path)
        # 复制图像
        for name in self.mix_data.train_names:
            ori_img_path = os.path.join(Path(self.data_root), Path("images_4"), Path(name))
            tar_img_path = os.path.join(Path(images_path), Path(name))
            shutil.copy(ori_img_path, tar_img_path)
        # 生成npy文件
        train_npy = os.path.join(Path(file_path), Path("poses_bounds.npy"))
        poses = self.llff_c2w_transform(self.mix_data.ori_c2w) # [N, 3, 4]
        hwf = self.generate_camera_hwf(self.mix_data.ori_intr) # [N, 3, 1]
        bounds = self.compute_depth_bounds(self.mix_data.all_pts, poses) # [N, 2]
        save_arr = torch.cat([poses, hwf], -1).reshape(poses.shape[0], -1) # [N, 15]
        save_arr = torch.cat([save_arr, bounds], -1) # [N, 17]
        print("save_arr: ", save_arr.shape)
        np.save(train_npy, save_arr.cpu().numpy())
        print('\nSave Param Finish !!!!')
    
    def copy_file(self, generate_dict, name):
        frames = generate_dict["frames"]
        folder_path = os.path.join(Path(self.save_path), Path(self.data_name), Path(name))
        # 创建文件夹
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)        
        # 逐帧复制图像
        for data in frames:
            img_name = data["file_path"].split("/")[-1]
            # 原始文件夹是train
            ori_img_path = os.path.join(Path(self.data_root), Path("images_4"), Path("{}.png".format(img_name)))
            tar_img_path = os.path.join(Path(folder_path), Path("{}.png".format(img_name)))
            shutil.copy(ori_img_path, tar_img_path)
        
        print('\nCopy Image Finish !!!!')

    # llff_c2ws:[N, 3, 4]
    def llff_c2w_transform(self, cv_pose):
        # pose_bottom = cv_pose[:, 3:, :].to(torch.float32)
        cv_c2ws = cv_pose[:, :3, :].to(torch.float32)
        # 从opencv坐标系转换到llff坐标系 [N, 3, 4]
        llff_c2ws = torch.cat([cv_c2ws[:, :, 1:2], cv_c2ws[:, :, 0:1], -cv_c2ws[:, :, 2:3], cv_c2ws[:, :, 3:4]], -1)
        return llff_c2ws
    
    # hwf:[N, 3, 1]
    def generate_camera_hwf(self, train_intr):
        # 取一个
        operate_intr = train_intr[0]
        cam_num = train_intr.shape[0]
        focal = operate_intr[0, 0]
        img_w = operate_intr[0, 2] * 2
        img_h = operate_intr[1, 2] * 2
        hwf = torch.tensor([img_h, img_w, focal]).reshape([3,1])
        hwf = hwf[None, :, :].repeat(cam_num, 1, 1)
        return hwf
    
    # all_bounds:[N, 2]
    def compute_depth_bounds(self, all_pts, poses):
        cam_num = all_pts.shape[0]
        all_pts = torch.tensor(all_pts, dtype = torch.float32)
        all_bounds = []
        for i in range(cam_num):
            pts = all_pts[i] # [H*W, 3]
            pose = poses[i]  # [3, 4]
            R = pose[:, :3]   # [3, 3]
            t = pose[:, 3]    # [3]
            pts_cam = (R @ (pts - t).T).T  # [H*W, 3]
            zs = -pts_cam[:, 2]
            # 剔除可能出现的负值或接近0的点（可选）
            zs = zs[zs > 0]
            if zs.numel() == 0:
                print("!!!!!!!!!!!!depth_error")
                close_depth = torch.tensor(0.0)
                inf_depth = torch.tensor(0.0)
            else:
                close_depth = torch.quantile(zs, 0.001)
                inf_depth = torch.quantile(zs, 0.999)
                print("close_depth, inf_depth: ", close_depth, inf_depth)
            bounds = torch.tensor([close_depth, inf_depth]).reshape([2])
            all_bounds.append(bounds)
        all_bounds = torch.stack(all_bounds, dim=0)
        print("all_bounds: ", all_bounds.shape)
        return all_bounds

### 数据命名中介
class Spann3r_to_mixdata():
    def __init__(self, spann3r_data):
        self.ori_c2w = spann3r_data.pca_c2w
        self.ori_w2c = self.inverse_RT(self.ori_c2w)
        self.ori_intr = spann3r_data.all_intr
        self.colors_gt = spann3r_data.all_color
        self.train_names = spann3r_data.name_order
        self.all_pts = spann3r_data.all_pts
    
    # 旋转矩阵
    def inverse_RT(self, rt_mat):
        # print(c2w_mat.shape)
        ori_R = rt_mat[:, :3, :3]
        ori_t = rt_mat[:, :3, 3:]
        ori_R_T = ori_R.transpose(-1, -2)
        # print(ori_R_T.shape, ori_t.shape)
        T_new = -ori_R_T @ ori_t
        T_w2c = torch.eye(4, device='cuda')[None, ...].repeat(rt_mat.shape[0], 1, 1)
        T_w2c[:, :3, :3] = ori_R_T
        T_w2c[:, :3, 3:] = T_new

        return T_w2c


        
if __name__ == "__main__":
    spann3r_data = Spann3r_Camera_data()
    mix_data = Spann3r_to_mixdata(spann3r_data)
    # trans_blender = Transform_spann3r_to_Blender(mix_data)
    trans_llff = Transform_spann3r_to_LLFF(mix_data)
