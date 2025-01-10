# GANs
WGAN,DCGAN,
GANs are used for image generation codes, mainly to expand image data through a game of generators and discriminators

##Software installation
1.Ububtu20.04
2.pytorch

###Python
1.$conda create -n torch python==3.8 
2.$conda activate torch
3.$conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
4.$conda install scikit-image

###RUN GANs
To train the GANs, simply run:
1.prepare your datasset
2.refine the main.py, include dataset, model, lr, sample_step
3.run train.py
4.the generated images saved at samples.

###License
The code related to the GAN algorithm is licensed under GNU Lesser General Public License v3.0. 


