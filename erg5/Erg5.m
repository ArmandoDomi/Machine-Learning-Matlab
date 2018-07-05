    fid = fopen('iris.data', 'r');
data = textscan(fid,'%f %f %f %f %s', 'Delimiter',','); %diavazw ta stoixeia apo to arxeio.
fclose(fid);  
NumberOfAttributes=length(data);                                                  
NumberOfPatterns=length(data{1});
x=zeros(NumberOfAttributes-1,NumberOfPatterns);
t=zeros(1,NumberOfPatterns);
class=zeros(1,NumberOfPatterns);
epilogi=0;
c_max=-1;
gamma_max=-1;
max_acc=-1;

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
 
 
%gamma=input('Dwse gamma = \n');
%C=input('Dwse C = \n');


while epilogi ~= 4
    
    fprintf('1.Diaxwrismos Iris-setosa apo Iris-virginica - Iris-versicolor\n');
    fprintf('2.Diaxwrismos Iris-versicolor apo Iris-setosa - Iris-virginica\n');
    fprintf('3.Diaxwrismos Iris-virginica apo Iris-setosa - Iris-versicolor\n');
    fprintf('4.exodos\n');
    
    
    
    epilogi=input('Dwse epilogi :\n');
    
    switch epilogi
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
            continue;
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
                option1='-s 0 ';
                flag=0;
            case 1
                option1='-s 1 ';
                flag=0;
            case 2
                option1='-s 2 ';
                flag=0;
            case 3
                option1='-s 3 ';
                flag=0;
            case 4
                option1='-s 4 ';
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
                    option2='-t 0 ';
                    flag=0;
                case 1
                    option2='-t 1 ';
                    flag=0;
                case 2
                    option2='-t 2 ';
                    flag=0;
                case 3
                    option2='-t 3 ';
                    flag=0;
                case 4
                    option2='-t 4 ';
                    flag=0;    
                otherwise
                    fprintf('Lathos epilogi...Dwse mia apo tis epiloges 1-4.\n')      
            end
             
    end    
    
    
    indices=crossvalind('Kfold',NumberOfPatterns,9);
    for gamma=0:0.1:1
        for C=1:10:101
            for i=1:9
        
                testidx=find(indices==i);
                trainidx=find(indices~=i);
       
                xtrain=x(:,trainidx);
                ttrain=t(trainidx);
                xtest=x(:,testidx);
                ttest=t(testidx);       
                ttrain1 = 2*ttrain - 1; 
                ttest1 = 2*ttest - 1; 
                options=sprintf('%s %s -g %i -c %i',option1,option2,gamma,C);
                %options=strcat(option1,option2,' -g ',gamma,' -c ',C);
                fprintf(options);
       
                model = svmtrain(ttrain1',xtrain',options);
       
                [predict_label,r_accuracy,prob_estimated]=svmpredict(ttest1',xtest',model,'-q');
                predict=predict_label>0; 
       
                accuracy(i)=evaluate(ttest',predict,'accuracy');
                precision(i)=evaluate(ttest',predict,'precision');
                recall(i)=evaluate(ttest',predict,'recall');
                fmeasure(i)=evaluate(ttest',predict,'fmeasure');
                sensitivity(i)=evaluate(ttest',predict,'sensitivity');
                specificity(i)=evaluate(ttest',predict,'specificity');
                
                %{
                subplot(3,3,i);
                plot(1:length(ttest),ttest,'b.');
                hold on;
                plot(1:length(predict),predict,'ro');
                hold off; 
                %}
            end
            
            if mean(accuracy) > max_acc 
                    max_acc=accuracy;
                    c_max=C;
                    gamma_max=gamma;
            end   
            
            fprintf('I mesi timi tou Accuracy gia ola ta folds einai : %f\n',mean(accuracy));
            fprintf('I mesi timi tou Precision gia ola ta folds einai : %f\n',mean(precision));
            fprintf('I mesi timi tou Recall gia ola ta folds einai : %f\n',mean(recall));
            fprintf('I mesi timi tou F-Measure gia ola ta folds einai : %f\n',mean(fmeasure));
            fprintf('I mesi timi tou sensitivity gia ola ta folds einai : %f\n',mean(sensitivity));
            fprintf('I mesi timi tou specificity gia ola ta folds einai : %f\n',mean(specificity));
            fprintf('\n');
            
        end    
    end
    
    
    fprintf('Max accuracy : %f\n',max(max_acc));
    fprintf('Max c_max : %f\n',c_max);
    fprintf('Max gamma_max : %f\n',gamma_max);
    fprintf('\n');


end


    
    
 
 