clear
close all
figure
fs_all = zeros(240,240,3);
for i =5:5
    i
    nii_t1_raw = load_untouch_nii(['./MRBrainS_raw/TrainingData/',num2str(i),'/T1.nii']);
    nii_t1IR_raw = load_untouch_nii(['./MRBrainS_raw/TrainingData/',num2str(i),'/T1_IR.nii']);
    nii_T2_raw = load_untouch_nii(['./MRBrainS_raw/TrainingData/',num2str(i),'/T2_FLAIR.nii']);
% load nii data
    
     nii_label_raw = load_untouch_nii(['./MRBrainS_raw/TrainingData/',num2str(i),'/LabelsForTraining.nii']);
    
    t1 = single(nii_t1_raw.img);
    t1IR = single(nii_t1IR_raw.img);
    t2 = single(nii_T2_raw.img); 
    
    label = nii_label_raw.img;

    for j =1:(size(t1,3)) %j slice index
        
        t1_ = t1(:,:,j);
        t1IR_ = t1IR(:,:,j);
        t2_ = t2(:,:,j);
        
        label_ = label(:,:,j);
        
        t1_ = rot90(t1_,3);
        t1IR_ = rot90(t1IR_,3);
        t2_ = rot90(t2_,3);
        label_ = rot90(label_,3);
        
         t1_ = (t1_ - min(t1_(:)))./(max(t1_(:)) - min(t1_(:)));%t1_./max(max(t1_));
         t1IR_ = (t1IR_ - min(t1IR_(:)))./(max(t1IR_(:)) - min(t1IR_(:)));  %t1IR_./max(max(t1IR_));
         t2_ = (t2_ - min(t2_(:)))./(max(t2_(:)) - min(t2_(:)));%t2_./max(max(t2_));
        
        label_(label_ ==1 | label_ ==2 )=1; %grey ... 4 kinds of label
        label_(label_ ==3 | label_ ==4  )=2;
        label_(label_ ==5 | label_ ==6 )=3;
        label_(label_ ==7 | label_ ==8 )=0; %background
        max(label_(:))
        
        
         fs_all(:,:,1) = t1_;
         fs_all(:,:,2) = t1IR_;
         fs_all(:,:,3) = t2_;
         
         save([ './data/MRbrain_mat_val/',num2str(i),'_',num2str(j),'_fs.mat'],'fs_all'); 
         save([ './data/MRbrain_mat_val/',num2str(i),'_',num2str(j),'_label.mat'],'label_'); 
         im_1 = [t1_,t1IR_,t2_];
         
        imwrite(abs(single(t1_)),[ './data/MRbrain_mat_val/',num2str(i),'_',num2str(j),'_t1.png'],'png')
        imwrite(abs(single(t1IR_)),[ './data/MRbrain_mat_val/',num2str(i),'_',num2str(j),'_t1IR.png'],'png')
        imwrite(abs(single(t2_)),[ './data/MRbrain_mat_val/',num2str(i),'_',num2str(j),'_t2.png'],'png')

    end
    
end