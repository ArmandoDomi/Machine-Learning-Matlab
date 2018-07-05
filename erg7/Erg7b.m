fid = fopen('Housing.txt', 'r');
data = textscan(fid,'%f %f %f %f %f %f %f %f %f %f %f %f %f %f', 'Delimiter','tab');
fclose(fid);                                            
NumberOfAttributes=length(data);                                                  
NumberOfPatterns=length(data{1});
x=zeros(NumberOfAttributes-1,NumberOfPatterns);
t=zeros(1,NumberOfPatterns);
best_n_mse=-1;
best_n_mae=-1;
min_Mse=10000;
min_Mae=10000;

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

    fprintf('1.tanh-Ypervoliki Efaptomeni\n');
    fprintf('2.Sigmoeidis\n');
    fprintf('3.Frammiki sunartisi\n');
    SinEnerKri=input('Epilogi sunartisi energopoihsis krifou strwmatos \n');
    switch SinEnerKri 
        case 1
            TF1='tansig';
            flag=0;
        case 2
            TF1='logsig';
            flag=0;
        case 3
            TF1='purelin';
            flag=0;
        otherwise
                fprintf('Lathos epilogi...Dwse mia apo tis epiloges 1-3.\n')      
    end
end
flag=1;
while flag
        
        fprintf('1.Aplo Back-Propagation \n');
        fprintf('2.Back-Propagation me ormi\n');
        fprintf('3.Back-Propagation ?? Conjugate Gradient\n');
        fprintf('4.Back-Propagation ?? Levenberg-Marquardt\n');
    
        mek=input('Epilogi methodou ekpetheusis BTF\n');
      
        switch mek
            
         case 1
            BTF='traingd';
            flag=0;
         case 2
            BTF='traingdm';
            flag=0;
         case 3
            BTF='traincgf';
            flag=0;
         case 4
            BTF='trainlm';
             flag=0;
         otherwise
            fprintf('Lathos epilogi...Dwse mia apo tis epiloges 1-4.\n')      
        end 
end
for N=5:5:50

    indices=crossvalind('Kfold',NumberOfPatterns,9);
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
       
       network=newff(x,t,N,{TF1 'purelin'}, BTF);
       net=train(network,xtrain,ttrain);
       simOut=sim(net,xtest);
       
       mse(i)=regrevaluateMSE(ttest,simOut);
       mae(i)=regrevaluateMAE(ttest,simOut);
       
       if mse(i) < min_Mse
           
          best_n_mse=N;
          
       end        
       if mae(i) < min_Mae
                
          best_n_mae=N;
           
       end 
       
    end
    
    fprintf('\n');
    fprintf('To  MSA gia ola ta folds einai : %f\n',mean(mse));
    fprintf('To MAE  gia ola ta folds einai : %f\n',mean(mae));
    fprintf('\n');
    
    
end
fprintf('H timi tou N pou dinoun to mikrotero MSE einai : N=%f \n',best_n_mse);
fprintf('H timi tou N pou dinoun to mikrotero MAE einai : N = %f \n',best_n_mae);


network=newff(x,t,best_n_mse,{TF1 'purelin'}, 'traincgf');
net=train(network,Xtrain,Ttrain);
simOut=sim(net,Xtest);

figure(1);
plot(1:length(Ttest),Ttest,'b.');
hold on;
plot(1:length(simOut),simOut,'ro');
hold off;
 
    
    
 
 