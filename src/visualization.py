from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import glfw
import numpy as np 
from PIL import Image
from PIL import ImageOps
from math import acos, cos, sin
from matplotlib import pyplot as plt
import time
import cv2
import yaml
from easydict import EasyDict as edict
from default_limbs import DEFAULT_SENSIBLE_LIMB_LENGTH, test_distance

STORECAMERADIR = None
IS_PERSPECTIVE = True  # 透視投影
VIEW = None
SCALE_K = None
ROTATE_K = None 
POS = None
LOOK_AT = None
ZOOM_IN = None
Y_PLANE = None
SLOPE = None 
EYE_UP = np.array([0.0, 1.0, 0.0]) # 定義對觀察者而言的上方（默認y軸的正方向）
WIN_W, WIN_H = 640, 480 # 保存窗口寬度和高度的變量
VIEW_W, VIEW_H = 640, 480
LEFT_IS_DOWNED = False # 滑鼠左鍵被按下
MOUSE_X, MOUSE_Y = 0, 0 # 考察滑鼠位移量時保存的起始位置
DIST = None
PHI = None
THETA = None
FRAME_ID = None
CAMERAS = None
PIDS = None
POSES = None
POSES_JOINTS_VIEWS = None
EYE = None
BONES = [
    [0, 1], [1, 2], [1, 3], [2, 4], [4, 6], [3, 5], [5, 7], [1, 8], [8, 9], [8, 10], [9, 11], [11, 13], [10, 12], [12, 14]
] # , [2, 9], [3, 10]
R = np.array ( [[1, 0, 0], [0, 0, 1], [0, -1, 0]] ) 

try:
    COLORS = np.array(plt.get_cmap('tab20').colors).astype(np.float32)[:, ::-1].tolist()
except AttributeError:  # if palette has not pre-defined colors
    COLORS = np.array(plt.get_cmap('tab20')(np.linspace(0, 1, 20))).astype(np.float32)[:, -2::-1].tolist()

def GetConfig(config_file):
	exp_config = None
	with open(config_file) as f:
		exp_config = edict(yaml.load(f, Loader=yaml.FullLoader))
	return exp_config

def setInfo(dataset, storecameradir, frame_id, cameras, shape, pids, poses, poses_joints_views):
    global PIDS, POSES, POSES_JOINTS_VIEWS
    global FRAME_ID, CAMERAS
    global WIN_W, WIN_H, VIEW_W, VIEW_H, R, EYE
    global STORECAMERADIR
    global DIST, PHI, THETA
    global VIEW, SCALE_K, ROTATE_K, POS, LOOK_AT, ZOOM_IN, Y_PLANE, SLOPE
    cfg = GetConfig('./configs/{}/vis_configs.yaml'.format(dataset))
    VIEW = np.array(cfg.VIEW) # 視景體的left/right/bottom/top/near/far六個面
    SCALE_K = np.array(cfg.SCALE_K) # 模型縮放比例 
    ROTATE_K = np.array(cfg.ROTATE_K)
    POS = np.array(cfg.POS) # 眼睛的位置（默認z軸的正方向） 
    LOOK_AT = np.array(cfg.LOOK_AT) # 瞄準方向的參考點（默認在坐標原點）
    ZOOM_IN = np.array(cfg.ZOOM_IN)
    Y_PLANE = cfg.Y_PLANE
    SLOPE = cfg.SLOPE

    STORECAMERADIR = storecameradir
    WIN_H = shape[0]
    WIN_W = (len(cameras) - 1) * (shape[1] + 10) + shape[1]
    VIEW_H, VIEW_W, _ = shape
    FRAME_ID = frame_id
    CAMERAS = [{'RT':camera.RT, 'P':camera.P} for camera in cameras]
    poses = [R @ i for i in poses.astype(np.float)]
    for i in range(len(poses_joints_views)):
        joints_views = np.zeros(17)
        for j, joints in enumerate(poses_joints_views[i]):
            joints_views[joints] = j
        poses_joints_views[i] = joints_views

    _poses = []
    _poses_joints_views = []
    for pose, joints_views in zip(poses, poses_joints_views):
        pose = np.transpose(pose)
        head = (pose[3] + pose[4]) / 2
        head_views = min(joints_views[3], joints_views[4])

        neck = (pose[5] + pose[6]) / 2
        neck_views = min(joints_views[5], joints_views[6])

        pelvis = (pose[11] + pose[12]) / 2
        pelvis_views = min(joints_views[11], joints_views[12])

        pose = np.append([head, neck], pose[5:]).reshape(-1, 3)
        pose = np.insert(pose, 8 * 3, pelvis).reshape(-1, 3)

        joints_views = np.append([head_views, neck_views], joints_views[5:])
        joints_views = np.insert(joints_views, 8, pelvis_views)

        _poses.append(pose)
        _poses_joints_views.append(joints_views)

    POSES = np.array(_poses)
    POSES_JOINTS_VIEWS = _poses_joints_views
    PIDS = pids
    EYE = [R @ np.linalg.inv(np.vstack([camera.RT, [0,0,0,1]])).dot(np.array([[0],[0],[0],[1]])).ravel()[:3] for camera in cameras]
    DIST, PHI, THETA = getposture()# 眼睛與觀察目標之間的距離、仰角、方位角 

def getposture(): 
    global EYE, LOOK_A 
    dist = np.sqrt(np.power((EYE-LOOK_AT), 2).sum()) 
    if dist > 0: 
        phi = np.arcsin((EYE[1]-LOOK_AT[1])/dist) 
        theta = np.arcsin((EYE[0]-LOOK_AT[0])/(dist*np.cos(phi))) 
    else: 
        phi = 0.0 
        theta = 0.0 
    return dist, phi, theta 

def init(): 
    glClearColor(1., 1., 1., 1.0) # 設置畫布背景色。注意：這裡必須是4個參數 
    glEnable(GL_DEPTH_TEST) # 開啟深度測試，實現遮擋關係 
    glDepthFunc(GL_LEQUAL) # 設置深度測試函數（GL_LEQUAL只是選項之一）

    light_ambient = (0.0, 0.0, 0.0, 1.0)
    light_diffuse = (1.0, 1.0, 1.0, 1.0)
    light_specular = (1.0, 1.0, 1.0, 1.0) 
    glLightfv(GL_LIGHT0, GL_POSITION, (10.0, 10.0, 0.0, 0.0))
    glLightfv(GL_LIGHT0, GL_AMBIENT , light_ambient)
    glLightfv(GL_LIGHT0, GL_DIFFUSE , light_diffuse)
    glLightfv(GL_LIGHT0, GL_SPECULAR, light_specular)

    glLightfv(GL_LIGHT1, GL_POSITION, (-10.0, 10.0, 0.0, 0.0))
    glLightfv(GL_LIGHT1, GL_AMBIENT , light_ambient)
    glLightfv(GL_LIGHT1, GL_DIFFUSE , light_diffuse)
    glLightfv(GL_LIGHT1, GL_SPECULAR, light_specular)

    glLightfv(GL_LIGHT2, GL_POSITION, (0.0, 10.0, 10.0, 0.0))
    glLightfv(GL_LIGHT2, GL_AMBIENT , light_ambient)
    glLightfv(GL_LIGHT2, GL_DIFFUSE , light_diffuse)
    glLightfv(GL_LIGHT2, GL_SPECULAR, light_specular)

    glLightfv(GL_LIGHT3, GL_POSITION, (0.0, 10.0, -10.0, 0.0))
    glLightfv(GL_LIGHT3, GL_AMBIENT , light_ambient)
    glLightfv(GL_LIGHT3, GL_DIFFUSE , light_diffuse)
    glLightfv(GL_LIGHT3, GL_SPECULAR, light_specular)

    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)
    glEnable(GL_LIGHT1)
    glEnable(GL_LIGHT2)
    glEnable(GL_LIGHT3)
    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
    glEnable(GL_COLOR_MATERIAL)

def drawGrids(length=100):
    global Y_PLANE, SLOPE
    glBegin(GL_LINES)
    for i in range(-length, length):
        glColor4f(0.5, 0.5, 0.5, 1.)
        glVertex3f(-length, i * SLOPE + Y_PLANE, i)
        glVertex3f(length, i * SLOPE + Y_PLANE, i)
        glColor4f(0.5, 0.5, 0.5, 1.)
        glVertex3f(i, length * SLOPE + Y_PLANE, length)
        glVertex3f(i, -length * SLOPE + Y_PLANE, -length)
    glEnd()

def drawAxis():
    global Y_PLANE, SLOPE
    glBegin(GL_LINES) # 開始繪製線段（世界坐標系） # 以紅色繪製x軸 
    glColor4f(1.0, 0.0, 0.0, 1.0) # 設置當前顏色為紅色不透明 
    glVertex3f(-100., Y_PLANE, 0.0) # 設置x軸頂點（x軸負方向） 
    glVertex3f(100., Y_PLANE, 0.0) # 設置x軸頂點（x軸正方向） # 以綠色繪製y軸 
    glColor4f(0.0, 1.0, 0.0, 1.0) # 設置當前顏色為綠色不透明 
    glVertex3f(0.0, -100., 0.0) # 設置y軸頂點（y軸負方向） 
    glVertex3f(0.0, 100., 0.0) # 設置y軸頂點（y軸正方向） # 以藍色繪製z軸 
    glColor4f(0.0, 0.0, 1.0, 1.0) # 設置當前顏色為藍色不透明 
    glVertex3f(0.0, -100 * SLOPE + Y_PLANE, -100.) # 設置z軸頂點（z軸負方向） 
    glVertex3f(0.0, 100 * SLOPE + Y_PLANE, 100.) # 設置z軸頂點（z軸正方向） 
    glEnd() # 結束繪製線段 

def drawJoint(pid, j):
    global PIDS, POSES
    x, y, z = POSES[pid, j]
    track_id = PIDS[pid]
    cx, cy, cz = COLORS[track_id % len(COLORS)]
    glColor4f(cz, cy, cx, 1.0)
    glTranslatef(x,y,z)
    if j == 0:
        glutSolidSphere(0.1, 20, 20)
    else:
        glutSolidSphere(0.04, 20, 20)

def drawBone(pid, b):
    global PIDS, POSES
    # if b == 7:
    #     track_id = PIDS[pid]
    #     cx, cy, cz = COLORS[track_id % len(COLORS)]
    #     glColor4f(cz, cy, cx, 1.0)
    #     pt1 = POSES[pid, 2]
    #     pt2 = POSES[pid, 3]
    #     pt3 = POSES[pid, 9]
    #     pt4 = POSES[pid, 10]
    #     drawBody(pid, pt1, pt2, pt3, pt4, 0.04)
    # else:
    pt1 = POSES[pid, BONES[b][0]]
    pt2 = POSES[pid, BONES[b][1]]
    if np.isnan(pt1).any() or np.isnan(pt2).any():
        return
    x, y, z = pt1
    length = np.linalg.norm(pt1-pt2)
    track_id = PIDS[pid]
    cx, cy, cz = COLORS[track_id % len(COLORS)]
    glColor4f(cz, cy, cx, 1.0)
    glTranslatef(x, y, z)

    v2r = pt2 - pt1
    z = np.array([0.0, 0.0, 1.0])
    ax = np.cross(z, v2r)
    l = np.sqrt(np.dot(v2r, v2r))
    angle = 180.0 / np.pi * acos(np.dot(z, v2r) / l)
    glRotatef(angle, ax[0], ax[1], ax[2])
    glutSolidCylinder(0.04, length, 20, 20)

def drawBody(pid, pt1, pt2, pt3, pt4, length):
    global PIDS
    track_id = PIDS[pid]
    pt1_norm = np.cross(pt3 - pt1, pt2 - pt1)
    pt2_norm = np.cross(pt1 - pt2, pt4 - pt2)
    pt3_norm = np.cross(pt4 - pt3, pt1 - pt3)
    pt4_norm = np.cross(pt2 - pt4, pt3 - pt4)
    pt1_1 = pt1 + pt1_norm / np.linalg.norm(pt1_norm) * length
    pt2_1 = pt2 + pt2_norm / np.linalg.norm(pt2_norm) * length
    pt3_1 = pt3 + pt3_norm / np.linalg.norm(pt3_norm) * length
    pt4_1 = pt4 + pt4_norm / np.linalg.norm(pt4_norm) * length
    pt1_2 = pt1 - pt1_norm / np.linalg.norm(pt1_norm) * length
    pt2_2 = pt2 - pt2_norm / np.linalg.norm(pt2_norm) * length
    pt3_2 = pt3 - pt3_norm / np.linalg.norm(pt3_norm) * length
    pt4_2 = pt4 - pt4_norm / np.linalg.norm(pt4_norm) * length
    
    cx, cy, cz = COLORS[track_id % len(COLORS)]
    glColor4f(cz, cy, cx, 1.0)
    glBegin(GL_TRIANGLES)
    glColor4f(cz, cy, cx, 1.0)
    glVertex3f(pt1_1[0], pt1_1[1], pt1_1[2])
    glVertex3f(pt2_1[0], pt2_1[1], pt2_1[2])
    glVertex3f(pt4_1[0], pt4_1[1], pt4_1[2])
    
    glColor4f(cz, cy, cx, 1.0)
    glVertex3f(pt1_1[0], pt1_1[1], pt1_1[2])
    glVertex3f(pt4_1[0], pt4_1[1], pt4_1[2])
    glVertex3f(pt3_1[0], pt3_1[1], pt3_1[2])

    glColor4f(cz, cy, cx, 1.0)
    glVertex3f(pt1_2[0], pt1_2[1], pt1_2[2])
    glVertex3f(pt2_2[0], pt2_2[1], pt2_2[2])
    glVertex3f(pt4_2[0], pt4_2[1], pt4_2[2])

    glColor4f(cz, cy, cx, 1.0)
    glVertex3f(pt1_2[0], pt1_2[1], pt1_2[2])
    glVertex3f(pt4_2[0], pt4_2[1], pt4_2[2])
    glVertex3f(pt3_2[0], pt3_2[1], pt3_2[2])
    glEnd()

def draw():
    global POSES, POSES_JOINTS_VIEWS, BONES, FRAME_ID
    glPushMatrix()
    drawGrids()
    glPopMatrix()

    glPushMatrix()
    drawAxis()
    glPopMatrix()
    for i in range(len(POSES)):
        # for j in range(len(POSES[i])):
        for j in range(1):
            # if POSES_JOINTS_VIEWS[i][j] == 0:
            #     print(FRAME_ID, i, j)
            #     continue
            glPushMatrix()
            drawJoint(i, j)
            glPopMatrix()

        for b in range(len(BONES)):
            # if test_distance(POSES[i, BONES[b][0]], POSES[i, BONES[b][1]], b): # POSES_JOINTS_VIEWS[i][BONES[b][0]] == 0 or POSES_JOINTS_VIEWS[i][BONES[b][1]] == 0 or 
            #     print(FRAME_ID, i, BONES[b])
            #     continue
            glPushMatrix()
            drawBone(i, b)
            glPopMatrix()

def display(): 
    global IS_PERSPECTIVE, VIEW 
    global EYE, LOOK_AT, EYE_UP
    global SCALE_K
    global WIN_W, WIN_H
    global FRAME_ID
    global STORECAMERADIR
    
    result = Image.new('RGBA', (WIN_W, WIN_H))
    
    # 設置投影（透視投影） 
    for idx, cam in enumerate(CAMERAS):
        # 清除螢幕及深度緩存 
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT) 

        glMatrixMode(GL_PROJECTION) 
        glLoadIdentity() 
        if VIEW_W > VIEW_H: 
            if IS_PERSPECTIVE: 
                glFrustum(VIEW[idx][0]*VIEW_W/VIEW_H, VIEW[idx][1]*VIEW_W/VIEW_H, VIEW[idx][2], VIEW[idx][3], VIEW[idx][4], VIEW[idx][5]) 
            else:
                glOrtho(VIEW[idx][0]*VIEW_W/VIEW_H, VIEW[idx][1]*VIEW_W/VIEW_H, VIEW[idx][2], VIEW[idx][3], VIEW[idx][4], VIEW[idx][5]) 
        else: 
            if IS_PERSPECTIVE: 
                glFrustum(VIEW[idx][0], VIEW[idx][1], VIEW[idx][2]*VIEW_H/VIEW_W, VIEW[idx][3]*VIEW_H/VIEW_W, VIEW[idx][4], VIEW[idx][5]) 
            else: 
                glOrtho(VIEW[idx][0], VIEW[idx][1], VIEW[idx][2]*VIEW_H/VIEW_W, VIEW[idx][3]*VIEW_H/VIEW_W, VIEW[idx][4], VIEW[idx][5]) # 設置模型視圖 

        glMatrixMode(GL_MODELVIEW) 
        glLoadIdentity() 
        glScale(SCALE_K[0], SCALE_K[1], SCALE_K[2])
        glRotatef(ROTATE_K[idx], 0, 0, 1)
        ORI = EYE[idx] + POS + ZOOM_IN[idx]
        gluLookAt( ORI[0], ORI[1], ORI[2], LOOK_AT[idx][0], LOOK_AT[idx][1], LOOK_AT[idx][2], EYE_UP[0], EYE_UP[1], EYE_UP[2] )
        glViewport(0, 0, VIEW_W, VIEW_H)
        draw()    
        glFlush()

        glPixelStorei(GL_PACK_ALIGNMENT, 4)
        data = glReadPixels(0, 0, VIEW_W, VIEW_H, GL_RGBA, GL_UNSIGNED_BYTE)
        image = Image.frombytes("RGBA", (VIEW_W, VIEW_H), data)
        image = ImageOps.flip(image) # in my case image is flipped top-bottom for some reason
        result.paste(image, ((VIEW_W + 10) * idx, 0))
        # image.save(STORECAMERADIR+os.sep+str(FRAME_ID)+'_'+str(idx)+'.png', 'PNG')
    result.save(STORECAMERADIR+os.sep+str(FRAME_ID)+'.png', 'PNG')

def plot3DPose(dataset, storecameraid, frame_id, cameras, shape, pids, poses, poses_joints_views):
    setInfo(dataset, storecameraid, frame_id, cameras, shape, pids, poses, poses_joints_views)
    glfw.init()
    glfw.window_hint(glfw.VISIBLE, False)
    window = glfw.create_window(VIEW_W, VIEW_H, "3D Pose", None, None)
    glfw.make_context_current(window)
    glutInit()
    init()
    display()
    glfw.destroy_window(window)
    glfw.terminate()


if __name__ == "__main__": 
    plot3DPose()