function dr = Dice_Ratio_3D(SEG, GT, structures)
  SEG = single(SEG==structures);
  GT = single(GT==structures);
  
  dr  = 2*nnz(SEG&GT)/(nnz(SEG) + nnz(GT));  
end