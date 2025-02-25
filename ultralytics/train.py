import warnings
from ultralytics import YOLO
warnings.filterwarnings('ignore')

if __name__=='__main__':
    model = YOLO(r'F:\github_repo\ultralytics-main\ultralytics\cfg\models\11\yolo11s-small.yaml')
    state_dict = model.model.state_dict()
    i = 0
    for (k1, v1) in state_dict.items():
        i = i + 1
        if i==50:
            break
        pass
        print(f'{i} Name1:{k1} Size: {v1.numel()}')
    # model.train(data=r"F:\ultralytics-main\data\VEDAI\data.yaml",
    #     cache=False,
    #     imgsz=640,
    #     epochs=1,
    #     single_cls=False,  # 是否是单类别检测
    #     batch=2,
    #     close_mosaic=0,
    #     workers=0,
    #     device='cpu',
    #     optimizer='SGD',  # using SGD 优化器 默认为auto建议大家使用固定的.
    #     amp=True,  # 如果出现训练损失为Nan可以关闭amp
    #     project='runs/train',
    #     name='exp',
    # )
