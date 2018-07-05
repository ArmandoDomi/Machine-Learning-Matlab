function value = evaluate( t,predict,criterion)
    
   
    
    tn = sum((t==0) & (predict==0));
    tp = sum((t==1) & (predict==1));
    fn = sum((t==1) & (predict==0));
    fp = sum((t==0) & (predict==1));
    

     switch criterion
        case 'accuracy'
            Accuracy=(tp+tn)/(tp+tn+fn+fp);
            value=Accuracy;
        case 'precision'
            Precision=tp/(tp+fp);
            value=Precision;
        case 'recall'
            Recall=tp/(tp+fn);
            value=Recall;
        case 'fmeasure'
            Fmeasure=((tp/(tp+fp))*(tp/(tp+fn)))/(((tp/(tp+fp))+(tp/(tp+fn)))/2);
            value=Fmeasure;
        case 'sensitivity'
            Sensitivity=tp/(tp+fn);
            value=Sensitivity;
        otherwise
            Specificity=tn/(tn+fp);
            value=Specificity;
 
    end
    
    
            
        
            
    
