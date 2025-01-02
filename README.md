This repository contains MATLAB reconstruction code for 3D stack-of-spiral real-time MRI. The code is being used in for the paper "***Three dimensional real-time MRI for the comprehensive evaluation of wrist kinematics***" that has been submitted to *Magnetic Resonance in Medicine*.

The script ```recon_rt_3d.m``` is the main reconstruction code. In the begining, several parameters are set, and followed by data sorting, sensitivity map estimation, and nonlinear conjugate gradient decent solver. The reconstruction can be run on GPU but a large GPU memory is required to store the internediate images and update terms. 

Representative raw data is shared at: https://drive.google.com/file/d/1je_l-JNo81lEoxeo55Q_Glka1hR3s9ww/view?usp=share_link. Please download the raw data, create a folder ```raw_data``` under the main diretory and place the downloaded data to the ```raw_data``` folder. 

contact: Ye Tian, phye1988@gmail.com
