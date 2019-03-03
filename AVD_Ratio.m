function dr = AVD_Ratio(SEG, GT)  

  dr  =  numel(find(SEG~=GT))/(240*240);
  dr = dr *100.;
end