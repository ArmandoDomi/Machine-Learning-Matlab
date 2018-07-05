fid = fopen('iris.data', 'r');
data = textscan(fid,'%f %f %f %f %s', 'Delimiter',','); %diavazw ta stoixeia apo to arxeio.
fclose(fid);                                             %kleinw to arxeio
NumberOfAttributes=length(data);                                                  
NumberOfPatterns=length(data{1});
x=zeros(NumberOfAttributes-1,NumberOfPatterns);
t=zeros(1,NumberOfPatterns);
class=zeros(1,NumberOfPatterns);
katwfli=1;
epilogi1=0;

 for i=1:NumberOfAttributes
    for j=1:NumberOfPatterns
        if i==5
            if strcmp('Iris-setosa',char(data{i}(j))) == 1
                class(j)=1;
            elseif strcmp('Iris-versicolor',char(data{i}(j))) == 1
                class(j)=2;
            else
                class(j)=3;
            end    
        else    
            x(i,j) = data{i}(j);
        end   
    end
 end
 
 while epilogi1 ~= 4
     
    fprintf('1.Diaxwrismos Iris-setosa apo Iris-virginica - Iris-versicolor\n');
    fprintf('2.Diaxwrismos Iris-versicolor apo Iris-setosa - Iris-virginica\n');
    fprintf('3.Diaxwrismos Iris-virginica apo Iris-setosa - Iris-versicolor\n');
    fprintf('4.exodos\n');
    
    epilogi1=input('Dwse epilogi :\n');
    
     switch epilogi1
        case 1
            t=class==1;
        case 2
            t=class==2;
        case 3
            t=class==3;
        case 4
            break;
        otherwise
            fprintf('Lathos epilogi...Dwse mia apo tis epiloges 1-4.\n');
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
         
        fprintf('1.tanh-Ypervoliki Efaptomeni\n');
        fprintf('2.Sigmoeidis\n');
        fprintf('3.Frammiki sunartisi\n');
    
        SinEnerExo=input('Epilogi sunartisi energopoihsis strwmatos e3odou \n');
        
        switch SinEnerExo 
         case 1
            TF2='tansig';
            flag=0;
         case 2
            TF2='logsig';
            flag=0;
         case 3
            TF2='purelin';
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
    
    
   indices=crossvalind('Kfold',NumberOfPatterns,9);
   
   for i=1:9
       
       testidx=find(indices==i);
       trainidx=find(indices~=i);
       xtrain=x(:,trainidx);
       ttrain=t(trainidx);
       xtest=x(:,testidx);
       ttest=t(testidx);
       Pltrain=length(xtrain);
       Pltest=length(xtest);
       Xtrain=[xtrain',ones(Pltrain,1)];
       Xtest=[xtest',ones(Pltest,1)];
       
       if strcmp(TF2,'tansig') == 1
            ttrain1 = 2*ttrain - 1;
            ttest1 = 2*ttest - 1;
            katwfli=0;
       elseif strcmp(TF2,'logsig') == 1
            ttrain1 = ttrain;
            ttest1 =  ttest;     
       else     
           ttrain1 = 2*ttrain - 1;
           ttest1 = 2*ttest - 1;
           katwfli=0;
       end
       
       
       N1=input('Dwse ton ari8mo twn krifwn neuronwn\n');
       
       network=newff(x,t,N1,{TF1 TF2}, BTF);
       
       net=train(network,xtrain,ttrain1);
       
       simOut=sim(net,xtest);
       
       if katwfli==1;
            predict=simOut>=0.5;
       else
            predict=simOut>=0;
       end
       
       
       
        accuracy(i)=evaluate(ttest,predict,'accuracy');
        precision(i)=evaluate(ttest,predict,'precision');
        recall(i)=evaluate(ttest,predict,'recall');
        fmeasure(i)=evaluate(ttest,predict,'fmeasure');
        sensitivity(i)=evaluate(ttest,predict,'sensitivity');
        specificity(i)=evaluate(ttest,predict,'specificity');
     
        subplot(3,3,i);
        plot(1:length(ttest),ttest,'b.');
        hold on;
        plot(1:length(predict),predict,'ro');
        hold off;    
       
       
   end 
   
   
    fprintf('I mesi timi tou Accuracy gia ola ta folds einai : %f\n',mean(accuracy));
    fprintf('I mesi timi tou Precision gia ola ta folds einai : %f\n',mean(precision));
    fprintf('I mesi timi tou Recall gia ola ta folds einai : %f\n',mean(recall));
    fprintf('I mesi timi tou F-Measure gia ola ta folds einai : %f\n',mean(fmeasure));
    fprintf('I mesi timi tou sensitivity gia ola ta folds einai : %f\n',mean(sensitivity));
    fprintf('I mesi timi tou specificity gia ola ta folds einai : %f\n',mean(specificity));
    fprintf('\n');
    
    
    
    
    
    
    
       
 end    