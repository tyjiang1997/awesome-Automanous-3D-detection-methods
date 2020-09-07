# awesome-Automanous-3D-detection-methods 
## note
This repositorie is created for anyone who wants to do research about 3D detection in automanous scence. We will update the lateset papers as soon as possible. If you find the awesome paper/code or have some suggestions, please contact [tianyjiang@gmail.com](tianyjiang@gmail.com)
### keywords
#### inputs
__`monocular`__: monocular &emsp; __`stereo`__: stereo &emsp;__`lidar`__: point cloud &emsp;</br>
__`image+lidar`__: image+lidar fusion &emsp;
#### datasets
experiments on datasets： __`kitti`__: KITTI &emsp; __`nuse`__: NuScenes &emsp; __`waymo`__: Waymo &emsp; __`ATG4D`__: ATG4D &emsp; __`lyft`__: lyft&emsp; 
#### code
framework ： __`Tensorflow`__: TensorFlow &emsp; __`PyTorch`__: PyTorch &emsp;


## 2017 
 - [[CVPR](http://openaccess.thecvf.com/content_cvpr_2017/papers/Qi_PointNet_Deep_Learning_CVPR_2017_paper.pdf)] PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation. [[tensorflow](https://github.com/charlesq34/pointnet)][[pytorch](https://github.com/fxia22/pointnet.pytorch)] [__`lidar`__] :fire::star:
 - [[CVPR](http://openaccess.thecvf.com/content_cvpr_2017/papers/Chen_Multi-View_3D_Object_CVPR_2017_paper.pdf)] Multi-View 3D Object Detection Network for Autonomous Driving. [[tensorflow](https://github.com/bostondiditeam/MV3D)] [__`image+lidar`__] [__`kitti`__]:fire: :star:
 - [[ICRA](https://ieeexplore.ieee.org/document/7989161)] Vote3Deep: Fast object detection in 3D point clouds using efficient convolutional neural networks. [[code_matlab](https://github.com/lijiannuist/Vote3Deep_lidar)] [__`lidar`__] [__`kitti`__]:star:
 - [[IROS](https://ieeexplore.ieee.org/document/8205955)] 3D fully convolutional network for vehicle detection in point cloud. [[tensorflow](https://github.com/yukitsuji/3D_CNN_tensorflow)] [__`lidar`__] [__`kitti`__]:fire: :star:
## 2018
 - [[CVPR](http://openaccess.thecvf.com/content_cvpr_2018/papers/Yang_PIXOR_Real-Time_3D_CVPR_2018_paper.pdf)] PIXOR: Real-time 3D Object Detection from Point Clouds. [[pytorch](https://github.com/ankita-kalra/PIXOR)]  [__`lidar`__] [__`kitti`__][__`ATG4D`__]
 - [[CVPR](http://openaccess.thecvf.com/content_cvpr_2018/papers/Zhou_VoxelNet_End-to-End_Learning_CVPR_2018_paper.pdf)] VoxelNet: End-to-End Learning for Point Cloud Based 3D Object Detection. [[tensorflow](https://github.com/tsinghua-rll/VoxelNet-tensorflow)] [__`lidar`__] [__`kitti`__]:fire::fire::fire: :star:
 - [[CVPR](http://openaccess.thecvf.com/content_cvpr_2018/papers/Xu_PointFusion_Deep_Sensor_CVPR_2018_paper.pdf)] PointFusion: Deep Sensor Fusion for 3D Bounding Box Estimation. [[code](https://github.com/malavikabindhi/CS230-PointFusion)] [__`image+lidar`__] [__`kitti`__]
 - [[CVPR](http://openaccess.thecvf.com/content_cvpr_2018/papers/Qi_Frustum_PointNets_for_CVPR_2018_paper.pdf)] Frustum PointNets for 3D Object Detection from RGB-D Data. [[tensorflow](https://github.com/charlesq34/frustum-pointnets)] [__`image+lidar`__] [__`kitti`__] :fire: :star:
 - [[ECCV](http://openaccess.thecvf.com/content_ECCV_2018/papers/Ming_Liang_Deep_Continuous_Fusion_ECCV_2018_paper.pdf)] Deep Continuous Fusion for Multi-Sensor 3D Object Detection. [__`image+lidar`__] [__`kitti`__] [__`ATG4D`__]
 - [[ECCVW](http://openaccess.thecvf.com/content_ECCVW_2018/papers/11131/Ali_YOLO3D_End-to-end_real-time_3D_Oriented_Object_Bounding_Box_Detection_from_ECCVW_2018_paper.pdf)] YOLO3D: End-to-end real-time 3D Oriented Object Bounding Box Detection from LiDAR Point Cloud. [ __`monocular`__] [__`kitti`__]
 - [[ICRA](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8462884)] End-to-end Learning of Multi-sensor 3D Tracking by Detection. [__`image+lidar`__] [__`kitti`__] 
 - [[ICRA](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8461257)] Robust Real-Time 3D Person Detection for Indoor and Outdoor Applications. [__`lidar`__] [__`kitti`__]
 - [[ICRA](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8461232)] A General Pipeline for 3D Detection of Vehicles.[__`lidar`__] [__`kitti`__]
 - [[IROS](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8594362)] Joint 3D Proposal Generation and Object Detection from View Aggregation. [__`lidar`__] [__`kitti`__]:star:
 - [[IROS](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8593910)] Edge and Corner Detection for Unorganized 3D Point Clouds with Application to Robotic Welding. [__`lidar`__] [__`kitti`__]
 - [[SENSORS](https://www.mdpi.com/1424-8220/18/10/3337)] SECOND: Sparsely Embedded Convolutional Detection. [[pytorch](https://github.com/traveller59/second.pytorch)][__`lidar`__] [__`kitti`__] :fire::fire::fire::fire:
 - [[arXiv](https://arxiv.org/abs/1812.05276)] IPOD: Intensive Point-based Object Detector for Point Cloud. [__`image+lidar`__] [__`kitti`__]
 - [[arXiv](https://arxiv.org/abs/1803.06199)] Complex-YOLO: Real-time 3D Object Detection on Point Clouds. [[pytorch](https://github.com/AI-liu/Complex-YOLO)]  [__`lidar`__] [__`kitti`__] :fire:
## 2019
 - [[CVPR](https://arxiv.org/abs/1812.07179)] Pseudo-LiDAR from Visual Depth Estimation: Bridging the Gap in 3D Object Detection for Autonomous Driving. [[code](https://github.com/mileyan/pseudo_lidar)] [__`stereo`__][__`kitti`__]
- [[CVPR](https://arxiv.org/pdf/1902.09738)]  Stereo R-CNN based 3D Object Detection for Autonomous Driving. [[code](https://github.com/HKUST-Aerial-Robotics/Stereo-RCNN)] [__`stereo`__][__`kitti`__]
 - [[CVPR](https://arxiv.org/abs/1812.04244)] PointRCNN: 3D Object Proposal Generation and Detection from Point Cloud. [[pytorch](https://github.com/sshaoshuai/PointRCNN)] [__`lidar`__] [__`kitti`__]:fire:
 - [[CVPR](https://arxiv.org/abs/1812.05784)] PointPillars: Fast Encoders for Object Detection from Point Clouds. [[pytorch](https://github.com/nutonomy/second.pytorch)] [__`lidar`__] [__`kitti`__]:fire:
 - [[CVPR](https://arxiv.org/abs/1903.08701v1)] LaserNet: An Efficient Probabilistic 3D Object Detector for Autonomous Driving.[__`lidar`__] [__`kitti`__][__`ATG4D`__]
 - [[CVPRW](http://openaccess.thecvf.com/content_CVPRW_2019/papers/WAD/Paigwar_Attentional_PointNet_for_3D-Object_Detection_in_Point_Clouds_CVPRW_2019_paper.pdf)] Attentional PointNet for 3D-Object Detection in Point Clouds. [[pytorch](https://github.com/anshulpaigwar/Attentional-PointNet)] [__`lidar`__] [__`kitti`__]
 - [[ICCV](https://arxiv.org/abs/1908.02990)] Fast Point R-CNN. [__`lidar`__] [__`kitti`__]
 - [[ICCV](https://arxiv.org/pdf/1907.10471)] STD: Sparse-to-Dense 3D Object Detector for Point Cloud.[[pytorch](https://github.com/tomztyang/3DSSD)] [__`lidar`__] [__`kitti`__]
 - [[ICCV](https://arxiv.org/pdf/1907.06038)] M3D-RPN: Monocular 3D Region Proposal Network for Object Detection.[[pytorch](http://cvlab.cse.msu.edu/project-m3d-rpn.html)] [__`monocular`__] [__`kitti`__]
 - [[ICCVW](https://arxiv.org/abs/1909.12249)] Range Adaptation for 3D Object Detection in LiDAR. [__`lidar`__] [__`kitti`__]
 - [[ICCVW](http://scholar.google.de/scholar?q=Multi-View%20Reprojection%20Architecture%20for%20Orientation%20Estimation)] Multi-View Reprojection Architecture for Orientation Estimation. [__`monocular`__] [__`kitti`__]
- [[NeurIPS](https://arxiv.org/pdf/1907.03739.pdf)] Point-Voxel CNN for Efficient 3D Deep Learning. [__`lidar`__] [__`kitti`__]
- [[ICMLW](https://arxiv.org/abs/1905.07290)] LiDAR Sensor modeling and Data augmentation with GANs for Autonomous driving. [__`lidar`__] 
- [[ICRA](https://arxiv.org/abs/1809.06065)] Focal Loss in 3D Object Detection. [[code](https://github.com/pyun-ram/FL3D)] [__`lidar`__] [__`kitti`__]
- [[ICRA](https://ras.papercept.net/conferences/conferences/ICRA19/program/ICRA19_ContentListWeb_2.html)] SEG-VoxelNet for 3D Vehicle Detection from RGB and LiDAR Data. [__`lidar`__] [__`kitti`__]
- [[ICRA](https://arxiv.org/abs/1904.01649)] MVX-Net: Multimodal VoxelNet for 3D Object Detection. [__`lidar`__] [__`kitti`__]
- [[AAAI](https://arxiv.org/pdf/1811.10247)] MonoGRNet: A Geometric Reasoning Network for 3D Object Localization. [__`monocular`__] [__`kitti`__]
- [[IROS](https://www.researchgate.net/publication/334720713_EPN_Edge-Aware_PointNet_for_Object_Recognition_from_Multi-View_25D_Point_Clouds)] EPN: Edge-Aware PointNet for Object Recognition from Multi-View 2.5D Point Clouds. [[tensorflow](https://github.com/Merium88/Edge-Aware-PointNet)] [__`lidar`__] [__`kitti`__]
- [[IROS](https://arxiv.org/pdf/1903.01864)] Frustum ConvNet: Sliding Frustums to Aggregate Local Point-Wise Features for Amodal 3D Object Detection. [[pytorch](https://github.com/zhixinwang/frustum-convnet)] [__`lidar+image`__] [__`kitti`__]
- [[IROS](https://arxiv.org/pdf/1907.06777)] Improving 3D object detection for pedestrians with virtual multi-view synthesis orientation estimation. [__`lidar`__] [__`kitti`__]
- [[3DV](https://arxiv.org/pdf/1908.03851)] IoU Loss for 2D/3D Object Detection. [__`lidar`__] [__`kitti`__]
- [[arXiv](https://arxiv.org/abs/1903.09847)] Monocular 3D Object Detection with Pseudo-LiDAR Point Cloud. [__`monocular`__][__`kitti`__]
- [[arXiv](https://arxiv.org/abs/1903.10750)] FVNet: 3D Front-View Proposal Generation for Real-Time Object Detection from Point Clouds. [[code](https://github.com/LordLiang/FVNet)] [__`lidar`__] [__`kitti`__]
- [[CVPRW](https://arxiv.org/abs/1904.07537)] Complexer-YOLO: Real-Time 3D Object Detection and Tracking on Semantic Point Clouds. [[pytorch](https://github.com/AI-liu/Complex-YOLO)] [__`monocular`__][__`kitti`__]:fire:
- [[CVPR](https://arxiv.org/pdf/1904.01690)] Monocular 3D Object Detection Leveraging Accurate Proposals and Shape Reconstruction. [[pytorch](https://github.com/kujason/monopsr)] [__`monocular`__][__`kitti`__]
- [[CVPR](https://arxiv.org/pdf/1903.10955.pdf)] GS3D: An Efficient 3D Object Detection Framework for Autonomous Driving.  [__`monocular`__][__`kitti`__]
- [[CVPR](https://arxiv.org/pdf/1812.02781.pdf)]  ROI-10D: Monocular Lifting of 2D Detection to 6D Pose and Metric Shape.  [__`monocular`__][__`kitti`__]
- [[CVPR](https://arxiv.org/pdf/1906.01193)] Triangulation Learning Network: from Monocular to Stereo 3D Object Detection. [[pytorch](https://github.com/Zengyi-Qin/TLNet)] [__`stereo`__][__`kitti`__]
- [[CoRR](https://arxiv.org/abs/1901.08373)] 3D Backbone Network for 3D Object Detection. [[code](https://github.com/Benzlxs/tDBN)] [__`lidar`__] [__`kitti`__]
- [[arXiv](https://arxiv.org/abs/1903.11027)] nuScenes: A multimodal dataset for autonomous driving. [[link](https://www.nuscenes.org/overview)] [__`dataset`__]
- [[arXiv](https://arxiv.org/pdf/1907.13079.pdf)] Deformable Filter Convolution for Point Cloud Reasoning.[__`lidar`__] [__`kitti`__][__`ATG4D`__]
- [[arXiv](https://arxiv.org/pdf/1911.12236)] PointRGCN: Graph Convolution Networks for 3D Vehicles Detection Refinement.[__`lidar`__] [__`kitti`__][__`ATG4D`__]
## 2020

 - [[TPAMI](https://arxiv.org/abs/1907.03670)] Part-A^2 Net: 3D Part-Aware and Aggregation Neural Network for Object Detection from Point Cloud. [[pytorch](https://github.com/open-mmlab/OpenPCDet)][__`lidar`__] [__`kitti`__]
- [[AAAI](https://arxiv.org/pdf/1912.05163.pdf)] TANet: Robust 3D Object Detection from Point Clouds with Triple Attention. [[code](https://github.com/happinesslz/TANet)] [__`lidar`__] [__`kitti`__]
- [[AAAI](https://arxiv.org/pdf/1911.06084)] PI-RCNN: An Efficient Multi-sensor 3D Object Detector with Point-based Attentive Cont-conv Fusion Module.  [__`lidar+image`__] [__`kitti`__]
- [[AAAI](https://arxiv.org/pdf/2003.00529)] ZoomNet: Part-Aware Adaptive Zooming Neural Network for 3D Object Detection.  [[code](https://github.com/detectRecog/ZoomNet)] [__`stereo`__] [__`kitti`__]
- [[AAAI](https://arxiv.org/pdf/2002.01619)] Monocular 3D Object Detection with Decoupled Structured Polygon Estimation and Height-Guided Depth Estimation.  [__`monocular`__] [__`kitti`__]
- [[CVPR](https://arxiv.org/pdf/1912.13192)] PV-RCNN: Point-Voxel Feature Set Abstraction for 3D Object Detection. [[pytorch](https://github.com/open-mmlab/OpenPCDet)] [__`lidar`__] [__`kitti`__] [__`waymo`__]:fire: :star: :fire: :star:
- [[CVPR](http://scholar.google.de/scholar?q=Structure%20Aware%20Single-stage%203D%20Object%20Detection%20from%20Point%20Cloud)] Structure Aware Single-stage 3D Object Detection from Point Cloud. [[pytorch](https://github.com/skyhehe123/SA-SSD)] [__`lidar`__] [__`kitti`__] :fire: :star: 
- [[CVPR](https://arxiv.org/pdf/2002.10187)]3DSSD: Point-based 3D Single Stage Object Detector. [[TensorFlow](https://github.com/tomztyang/3DSSD)] [__`lidar`__] [__`kitti`__][__`nusc`__] :fire: :star: 
- [[CVPR](https://arxiv.org/pdf/2003.01251)]Point-GNN: Graph Neural Network for 3D Object Detection in a Point Cloud. [[TensorFlow](https://github.com/WeijingShi/Point-GNN)] [__`lidar`__] [__`kitti`__] :fire: :star: 
- [[CVPR](https://arxiv.org/pdf/2006.04356)]Associate-3Ddet: Perceptual-to-Conceptual Association for 3D Point Cloud Object Detection.  [__`lidar`__] [__`kitti`__] 
- [[CVPR](http://openaccess.thecvf.com/content_CVPR_2020/papers/Liang_PnPNet_End-to-End_Perception_and_Prediction_With_Tracking_in_the_Loop_CVPR_2020_paper.pdf)]PnPNet: End-to-End Perception and Prediction with Tracking in the Loop.  [__`lidar`__] 
- [[CVPR](http://openaccess.thecvf.com/content_CVPR_2020/papers/Wang_Train_in_Germany_Test_in_the_USA_Making_3D_Object_CVPR_2020_paper.pdf)] Train in Germany, Test in The USA: Making 3D Object Detectors Generalize.[[code](https://github.com/cxy1997/3D_adapt_auto_driving)] [__`lidar`__] 
- [[CVPR](https://arxiv.org/pdf/1911.10150)] PointPainting: Sequential Fusion for 3D Object Detection. [__`lidar+image`__] [__`kitti`__] [__`nusc`__] 
- [[CVPR](https://arxiv.org/pdf/2001.03398)]  DSGN: Deep Stereo Geometry Network for 3D Object Detection. [__`monocular`__] [__`kitti`__] 
- [[CVPR](https://arxiv.org/pdf/2004.03572)]  Disp R-CNN: Stereo 3D Object Detection via Shape Prior Guided Instance Disparity Estimation.[[code](https://github.com/zju3dv/disprcnn)] [__`stereo`__] [__`kitti`__] 
- [[CVPR](https://arxiv.org/pdf/1912.04799)] Learning Depth-Guided Convolutions for Monocular 3D Object Detection.[[code](https://github.com/dingmyu/D4LCN)] [__`monocular`__] [__`kitti`__] 
- [[CVPR](https://arxiv.org/pdf/2003.00504)] MonoPair: Monocular 3D Object Detection Using Pairwise Spatial Relationships. [__`monocular`__] [__`kitti`__] 
- [[CVPR](https://arxiv.org/pdf/2004.01389)] LiDAR-based Online 3D Video Object Detection with Graph-based Message Passing and Spatiotemporal Transformer Attention. [__`lidar_video`__] [__`nusc`__] 
- [[CVPR](https://arxiv.org/pdf/2004.00543)] Physically Realizable Adversarial Examples for LiDAR Object Detection. [__`lidar`__] 
- [[CVPR](https://arxiv.org/pdf/2003.00186)]HVNet: Hybrid Voxel Network for LiDAR Based 3D Object Detection. [__`lidar`__] [__`kitti`__] 
- [[CVPR](https://arxiv.org/pdf/2004.08745)]Learning to Evaluate Perception Models Using Planner-Centric Metrics. [__`lidar`__]
- [[CVPR](https://arxiv.org/pdf/1912.04986)]What You See is What You Get: Exploiting Visibility for 3D Object Detection. [__`lidar`__] [__`nusc`__] 
- [[CVPR](https://arxiv.org/pdf/2003.06754)]MotionNet: Joint Perception and Motion Prediction for Autonomous Driving Based on Bird’s Eye View Maps. [__`lidar`__] 
- [[ECCVW](https://arxiv.org/pdf/2008.08766)]  Deformable PV-RCNN: Improving 3D Object Detection with Learned Deformations.[[code](https://github.com/AutoVision-cloud/Deformable-PV-RCNN)][__`lidar`__] [__`kitti`__] 
- [[ECCV](http://scholar.google.de/scholar?q=object%20as%20hotspots)]  object as hotspots.[__`lidar`__] [__`kitti`__] 
- [[ECCV](https://arxiv.org/pdf/2007.08856)]   EPNet: Enhancing Point Features with Image Semantics for 3D Object Detection.[__`lidar+image`__] [__`kitti`__] 
- [[ECCV](https://arxiv.org/pdf/2004.12636)]   3D-CVF: Generating Joint Camera and LiDAR Features Using Cross-View Spatial Feature Fusion for 3D Object Detection.[__`lidar+image`__] [__`kitti`__] 
- [[ECCV](https://arxiv.org/pdf/2007.09548)]    Kinematic 3D Object Detection in Monocular Video.[[code](http://cvlab.cse.msu.edu/project-kinematic.html)][__`monocular_video`__] [__`kitti`__] 
- [[ECCV](https://arxiv.org/pdf/2008.04582)]   Rethinking Pseudo-LiDAR Representation.[[code](https://github.com/xinzhuma/patchnet)][__`monocular`__] [__`kitti`__] 
- [[ECCV](https://arxiv.org/pdf/2007.12392)]  An LSTM Approach to Temporal 3D Object Detection in LiDAR Point Clouds.[__`lidar`__] [__`kitti`__] 
- [[ECCV](https://arxiv.org/pdf/2007.10323)]  Pillar-based Object Detection for Autonomous Driving.[__`lidar`__] [__`waymo`__] 
- [[ECCV](https://arxiv.org/pdf/2008.02191.pdf)]  Active Perception using Light Curtains for Autonomous Driving.[[code](http://siddancha.github.io/projects/active-perception-light-curtains)][__`lidar`__] 
- [[ECCV](https://arxiv.org/pdf/2007.16100.pdf)]  Searching Efficient 3D Architectures with Sparse Point-Voxel Convolution.[__`lidar`__] 
- [[ECCV](https://arxiv.org/pdf/2004.00831)]  Improving 3D Object Detection through Progressive Population Based Augmentation.[__`lidar`__] [__`kitti`__] 
- [[IROS](https://arxiv.org/pdf/2006.05518)] MVLidarNet: Real-Time Multi-Class Scene Understanding for Autonomous Driving Using Multiple Views.[__`lidar`__] [__`nusc`__] 
- [[ACMMM](https://arxiv.org/pdf/2007.13970.pdf)] Weakly Supervised 3D Object Detection from Point Clouds.[__`lidar`__]
- [[BMVC](https://arxiv.org/pdf/2005.10863)] RV-FuseNet: Range View based Fusion of Time-Series LiDAR Data for Joint 3D Object Detection and Motion Forecasting [__`lidar`__][__`nusc`__]
- [[Sensors]()] 3D-GIoU: 3D Generalized Intersection over Union for Object Detection in Point Cloud [__`lidar`__][__`kitti`__]
- [[arxiv](https://arxiv.org/pdf/2004.04962)] 3D IoU-Net: IoU Guided 3D Object Detector for Point Clouds [__`lidar`__][__`kitti`__]
- [[arxiv](https://arxiv.org/pdf/2006.11275)] Center-based 3D Object Detection and Tracking [[code](https://github.com/tianweiy/CenterPoint)][__`lidar`__][__`nusc`__]
- [[arxiv](https://arxiv.org/pdf/2004.00186)] Boundary-Aware Dense Feature Indicator for Single-Stage 3D Object Detection from Point Clouds [__`lidar`__][__`nusc`__]
- [[arxiv](https://arxiv.org/pdf/2007.08556)] InfoFocus: 3D Object Detection for Autonomous Driving with Dynamic Information Modeling [__`lidar`__][__`nusc`__]
- [[arxiv](https://arxiv.org/pdf/2004.01643)] Quantifying Data Augmentation for LiDAR based 3D Object Detection [__`lidar`__][__`kitti`__]
- [[arxiv](https://arxiv.org/pdf/2005.09927)] Range Conditioned Dilated Convolutions for Scale Invariant 3D Object Detection [__`lidar`__][__`kitti`__][__`nusc`__]
- [[arxiv](https://arxiv.org/pdf/2003.10670)] Real-time 3D object proposal generation and classification under limited processing resources [__`lidar`__][__`kitti`__]
- [[arxiv](https://arxiv.org/pdf/2003.11242)] Safety-Aware Hardening of 3D Object Detection Neural Network Systems [__`lidar`__][__`kitti`__]
- [[arxiv](https://arxiv.org/pdf/2006.05187)] Stereo RGB and Deeper LIDAR Based Network for 3D Object Detection[__`stereo`__][__`kitti`__]
- [[arxiv](https://arxiv.org/pdf/2004.02774)] SSN: Shape Signature Networks for Multi-class Object Detection from Point Clouds [[code](https://github.com/xinge008/SSN)][__`lidar`__][__`kitti`__]
- [[arxiv](https://arxiv.org/pdf/2006.04043)] SVGA-Net: Sparse Voxel-Graph Attention Network for 3D Object Detection from Point Clouds
[__`lidar`__][__`kitti`__]
- [[arxiv](https://arxiv.org/pdf/2008.12008.pdf)] GhostBuster: Looking Into Shadows to Detect Ghost Objects in Autonomous Vehicle 3D Sensing [__`lidar`__][__`kitti`__]
- [[arxiv](https://arxiv.org/pdf/2008.10436.pdf)] Cross-Modality 3D Object Detection [__`lidar`__][__`kitti`__]
- [[arxiv](https://arxiv.org/pdf/2008.09672)] Towards Autonomous Driving: a Multi-Modal 360∘ Perception Proposal[__`lidar`__][__`kitti`__]
- [[arxiv](https://arxiv.org/pdf/2009.00206)]  RangeRCNN: Towards Fast and Accurate 3D Object Detection with Range Image Representation[__`lidar`__][__`kitti`__]
- [[arxiv](https://arxiv.org/pdf/2009.00784.pdf)] CLOCs: Camera-LiDAR Object Candidates Fusion for 3D Object Detection[__`lidar+image`__][__`kitti`__]
- [[arxiv](https://arxiv.org/pdf/2008.13748)]Reinforced Axial Refinement Network for Monocular 3D Object Detection[__`monocular`__][__`kitti`__]
- [[arxiv](https://arxiv.org/pdf/2008.12599.pdf)]PV-RCNN: The Top-Performing LiDAR-only Solutions for 3D Detection / 3D Tracking / Domain Adaptation of Waymo Open Dataset Challenges[__`lidar`__][__`kitti`__][__`waymo`__]
- [[arxiv](https://arxiv.org/pdf/2008.12008.pdf)]GhostBuster: Looking Into Shadows to Detect Ghost Objects in Autonomous Vehicle 3D Sensing [__`lidar`__][__`kitti`__]
- [[arxiv](https://arxiv.org/pdf/2007.07214)]CenterNet3D:An Anchor free Object Detector for Autonomous Driving [__`lidar`__][__`kitti`__]

- [[arxiv](https://arxiv.org/pdf/2006.15505.pdf)] 1st Place Solution for Waymo Open Dataset Challenge - 3D Detection and Domain Adaptation [__`lidar`__][__`waymo`__]

## Suvery
- [[TPAMI](https://arxiv.org/pdf/1912.12033)] Deep Learning for 3D Point Clouds: A Survey[__`lidar`__]
 - [[arxiv](https://arxiv.org/pdf/2003.00601)] 3D Point Cloud Processing and Learning for Autonomous Driving[__`lidar`__]
## code base 
- [[lidar_only](https://github.com/traveller59/second.pytorch)] second.pytorch [__`kitti`__][__`nusc`__]
- [[lidar_only](https://github.com/poodarchu/Det3D)] Det3D [__`kitti`__][__`nusc`__][__`lyft`__][__`waymo`__]
- [[lidar_only](https://github.com/open-mmlab/OpenPCDet)] OpenPCDet[__`kitti`__][__`nusc`__][__`waymo`__]
 - [[lidar_image](https://github.com/open-mmlab/mmdetection3d)] mmdetection3d[__`kitti`__][__`nusc`__][__`lyft`__][__`waymo`__]
