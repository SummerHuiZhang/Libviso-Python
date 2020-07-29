import pyviso as viso
import sys
import numpy as np
import matplotlib.pyplot as plt
import cv2

def main():
    viso_mono = viso.VisoMomo(718.856,607.1928,185.2157)
    images_path = sys.argv[1]
    tag = '.test_001'
    if len(sys.argv)>2:
        tag = sys.argv[2]
    #    real_scale = np.loadtxt(sys.argv[2])
    res_addr = 'result/'+images_path.split('.')[-2].split('/')[-1]+'_'
    print(images_path.split('.')[-2].split('/')[-1])
    images      = open(images_path)
    image_name = images.readline() # first line is not pointing to a image, so we read and skip it
    image_names= images.read().split('\n')
    
    motions=[]
    feature2ds = []
    feature3ds = []
    move_flags = []
    path = []
    scales=[]
    frame_id = 0
    img_last = np.zeros((360,1200))
    for image_name in image_names:
        if frame_id<0:
            frame_id+=1
            continue
        print(image_name)
        if len(image_name) == 0:
            break
        move_flag = viso_mono.process(image_name)
        img = cv2.imread(image_name)
        motion = np.array(viso_mono.get_motion())
        motion_trans = motion[3:12:4]
        square = np.sum(motion_trans*motion_trans)
        speed = np.sqrt(square)
        motion_trans = motion_trans/(speed+0.000000001)
        motion[3:12:4] = motion_trans

        motions.append(np.array(motion))
        scales.append(speed)
        feature3d = np.array(viso_mono.get_feature3d())
        feature2d = np.array(viso_mono.get_feature3d()).copy()
        if feature2d.shape[0]>0:
            feature2d[:,0] = feature2d[:,0]*718.856/feature2d[:,2]+607.1928
            feature2d[:,1] = feature2d[:,1]*718.856/feature2d[:,2]+185.2157
            feature2d= feature2d[:,0:2]
        feature2ds.append(feature2d)
        feature3ds.append(feature3d)
        move_flags.append(move_flag)
        if False: #frame_id%4000==10: #move_flag == False:
            path = get_path(np.array(motions),1)
            print(feature2d.shape)
            draw_feature(img_last,np.array(viso_mono.get_feature2d()))
            #draw_feature(img,feature2d)
            plt.imshow(img_last)
            #plt.plot(path[:,3],path[:,11])
            plt.show()
        frame_id+=1
        img_last = img.copy()
    data_to_save = {}
    data_to_save['motions'] = motions[1:]
    print(len(motions))
    data_to_save['feature3ds'] = feature3ds[1:]
    data_to_save['feature2ds'] = feature2ds[1:]
    data_to_save['move_flags'] = move_flags[1:]
    np.save(res_addr+tag+'result.npy',data_to_save)
    path = get_path(np.array(motions),scales)
    np.savetxt(res_addr+tag+'path.txt',path[1:])
    np.savetxt(res_addr+tag+'scales.txt',scales[1:])
def line2mat(line_data):
    mat = np.eye(4)
    mat[0:3,:] = line_data.reshape(3,4)
    return np.matrix(mat)

def motion2pose(data):
    data_size = data.shape[0]
    all_pose = np.zeros((data_size+1,12))
    temp = np.eye(4,4).reshape(1,16)
    all_pose[0,:] = temp[0,0:12]
    pose = np.matrix(np.eye(4,4))
    for i in range(0,data_size):
        data_mat = line2mat(data[i,:])
        pose = pose*data_mat
        pose_line = np.array(pose[0:3,:]).reshape(1,12)
        all_pose[i+1,:] = pose_line
    return all_pose


def get_path(motions,scales):
    motion_trans = motions[:,3:12:4]
    motion_trans = (scales*motion_trans.transpose()).transpose()
    motions[:,3:12:4] = motion_trans
    pose = motion2pose(motions)
    return pose


def draw_feature(img,feature,color=(255,255,0)):
    for i in range(feature.shape[0]):
        cv2.circle(img,(int(feature[i,0]),int(feature[i,1])),3,color,-1)

if __name__ =='__main__':
    main()
