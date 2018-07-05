fid = fopen('Housing.txt', 'r');
data = textscan(fid,'%f %f %f %f %f %f %f %f %f %f %f %f %f %f', 'Delimiter','tab'); 
fclose(fid);  
NumberOfAttributes=length(data);                                                  
NumberOfPatterns=length(data{1});
x=zeros(NumberOfAttributes-1,NumberOfPatterns);
t=zeros(1,NumberOfPatterns);
mse=zeros(1,9);
mae=zeros(1,9);
best_c_mse=-1;
best_gamma_mse=-1;
best_c_mae=-1;
best_gamma_mae=-1;
min_Mse=10000;
min_Mae=10000;
m_mse=0;
m_mae=0;

 for i=1:NumberOfAttributes-1
    for j=1:NumberOfPatterns
        x(i,j)=data{i}(j);
    end
 end
 
 for i=1:NumberOfPatterns
    t(i)=data{14}(i);
 end
 
  flag=1;
  while flag   
      
         fprintf('0 -- C-SVC \n');
         fprintf('1 -- nu-SVC\n');
         fprintf('2 -- one-class SVM\n');
         fprintf('3 -- epsilon-SVR\n');
         fprintf('4 -- nu-SVR\n');
          
         svm_type=input('Dwse epilogi\n'); 
         switch svm_type
            
            case 0
                option1='-s 0';
                flag=0;
            case 1
                option1='-s 1';
                flag=0;
            case 2
                option1='-s 2';
                flag=0;
            case 3
                option1='-s 3';
                flag=0;
            case 4
                option1='-s 4';
                flag=0;    
             otherwise
                fprintf('Lathos epilogi...Dwse mia apo tis epiloges 1-4.\n')      
         end     
  end
  
  flag=1;
  while flag
                    
    fprintf('0 -- linear:u''*v\n'); 
    fprintf('1 -- polynomial:(gamma*u''*v+coef0)^degree \n');
    fprintf('2 -- radial basis function:exp(-gamma*|u-v|^2)\n');
    fprintf('3 -- sigmoid:tanh(gamma*u''*v+coef0)\n');
    fprintf('4 -- precomputed kernel(kernel values in training_instance_matrix)\n');
              
    kernel_type=input('Dwse epilogi\n'); 
    switch kernel_type
        case 0
            option2='-t 0';
            flag=0;
        case 1
            option2='-t 1';
            flag=0;
        case 2
            option2='-t 2';
            flag=0;
        case 3
            option2='-t 3';
            flag=0;
        case 4
            option2='-t 4';
            flag=0;    
        otherwise
            fprintf('Lathos epilogi...Dwse mia apo tis epiloges 1-4.\n')      
    end
             
  end 
 
 
 
 indices=crossvalind('Kfold',NumberOfPatterns,9);

 gamma=0.0001;
 C=1;
 while gamma < 1
    while C < 10000
       for i=1:9
            
           
            if i==1
                Testidx=find(indices==i);
                Trainidx=find(indices~=i);       
                Xtrain=x(:,Trainidx);
                Ttrain=t(Trainidx);
                Xtest=x(:,Testidx);
                Ttest=t(Testidx);
            end
            
            testidx=find(indices==i);
            trainidx=find(indices~=i);       
            xtrain=x(:,trainidx);
            ttrain=t(trainidx);
            xtest=x(:,testidx);
            ttest=t(testidx);
           
            options=sprintf('%s %s -g %f -c %f',option1,option2,gamma,C);
            
            model = svmtrain(ttrain',xtrain',options);
            
            [predict_reg,r_accuracy,prob_estimated]=svmpredict(ttest',xtest',model,'-q');
            
            mse(i)=regrevaluateMSE(ttest',predict_reg);
            
            mae(i)=regrevaluateMAE(ttest',predict_reg);
         
            if mse(i) < min_Mse
                
                best_c_mse=C;
                best_gamma_mse=gamma;
                m_mse=m_mse+1;
            end
            
            if mae(i) < min_Mae
                
                best_c_mae=C;
                best_gamma_mae=gamma;
                m_mae=m_mae+1;
            end    
       end
       fprintf('\n');
       fprintf('To  MSA gia ola ta folds einai : %f\n',mean(mse));
       fprintf('To MAE  gia ola ta folds einai : %f\n',mean(mae));
       fprintf('\n');
       
        C=C*10;     
    end
    gamma=gamma*10;
 end
 
 fprintf('Oi times tou gamma kai c pou dinoun to mikrotero MSE einai : gamma = %f , C=%f \n',best_gamma_mse,best_c_mse);
 fprintf('Oi times tou gamma kai c pou dinoun to mikrotero MAE einai : gamma = %f , C=%f \n',best_gamma_mae,best_c_mae);
 
 %mono to fold 1 
 
 
 options=sprintf('%s %s -g %f -c %f',option1,option2,best_gamma_mse,best_c_mse);
 %options=sprintf('%s %s -g %f -c %f',option1,option2,0.1,10);
 
 model = svmtrain(Ttrain',Xtrain',options);
 [predict_reg]=svmpredict(Ttest',Xtest',model,'-q');
 figure(1);
 plot(1:length(Ttest),Ttest,'b.');
 hold on;
 plot(1:length(predict_reg),predict_reg,'ro');
 hold off; 

 
 
 
 
 
 