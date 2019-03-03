clear
close all
clc
min_ = 1;
max_ = 48;
logit = zeros(240,240,48);
label = zeros(240,240,48);

dice_coes = zeros(max_-min_+1,1);

AVD_coes1 = zeros(max_-min_+1,1);
AVD_coes2 = zeros(max_-min_+1,1);
AVD_coes3 = zeros(max_-min_+1,1);
 
MHD_coes1 = zeros(max_-min_+1,1);
MHD_coes2 = zeros(max_-min_+1,1);
MHD_coes3 = zeros(max_-min_+1,1);

path_imag = 'tea_stu_encoderftune_1D30_MRBrain_enlarge_BN';
for i =min_:max_
    i
    logit_ = load([ '.\test\',path_imag,'\5_',num2str(i),'_fs_logit','.mat']);
%     figure(),imagesc(logit);colormap jet;axis off;
    logit(:,:,i) = logit_.logit;    
    
    label_ = load(['.\test\',path_imag,'\5_',num2str(i),'_fs_label.mat']);
%     figure(),imagesc(label);colormap jet;axis off;
    label(:,:,i) = label_.label;
    
    MHD_coes1(i-min_+1)  = ModHausdorffDist(single(logit(:,:,i)==1),single(label(:,:,i)==1));
    MHD_coes2(i-min_+1)  = ModHausdorffDist(single(logit(:,:,i)==2),single(label(:,:,i)==2));
    MHD_coes3(i-min_+1)  = ModHausdorffDist(single(logit(:,:,i)==3),single(label(:,:,i)==3));
    
    AVD_coes1(i-min_+1)  = AVD_Ratio(single(logit(:,:,i)==1),single(label(:,:,i)==1));
    AVD_coes2(i-min_+1)  = AVD_Ratio(single(logit(:,:,i)==2),single(label(:,:,i)==2));
    AVD_coes3(i-min_+1)  = AVD_Ratio(single(logit(:,:,i)==3),single(label(:,:,i)==3));
end

dice_coes1  = Dice_Ratio_3D(logit, label, 1);  
dice_coes2  = Dice_Ratio_3D(logit, label, 2); 
dice_coes3  = Dice_Ratio_3D(logit, label, 3);  

'mean dice_coes:'
mean([mean(dice_coes1), mean(dice_coes2), mean(dice_coes3)])
'mean MHD_coes:'
mean([mean(MHD_coes1), mean(MHD_coes2), mean(MHD_coes3)])
'mean AVD_coes:'
mean([mean(AVD_coes1), mean(AVD_coes2), mean(AVD_coes3)])
