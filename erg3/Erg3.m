fid = fopen('iris.data', 'r');
data = textscan(fid,'%f %f %f %f %s', 'Delimiter',','); %diavazw ta stoixeia apo to arxeio.
fclose(fid);                                             %kleinw to arxeio
NumberOfAttributes=length(data);                                                  
NumberOfPatterns=length(data{1});
x=zeros(NumberOfAttributes-1,NumberOfPatterns);
t=zeros(1,NumberOfPatterns);
class=zeros(1,NumberOfPatterns);

epilogi=0;

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

 while epilogi~=4
     
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
    end
    %zitame apo ton xristi na eisagei to plithos twn epoxwn
    epoxes=input('Dwse to plithos twn epoxwn \n');

    %zitame apo ton xristi na eisagei to vima ekpetheusis
    b=input('Dwse to vima ekpedeusis \n');
    
    indices=crossvalind('Kfold',NumberOfPatterns,9);
    figure(1);
    
    
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
    
        
        ttrain1 = 2*ttrain - 1; % metatropi twn 0 se -1 kai to 1 paramenei 1
        ttest1 = 2*ttest - 1;   % metatropi twn 0 se -1 kai to 1 paramenei 1
        
        w=perceptron(Xtrain',ttrain1,epoxes,b);
        fprintf('\n');
        y=Xtest*w';
        predict=y>0;
        
        accuracy(i)=evaluate(ttest',predict,'accuracy');
        precision(i)=evaluate(ttest',predict,'precision');
        recall(i)=evaluate(ttest',predict,'recall');
        fmeasure(i)=evaluate(ttest',predict,'fmeasure');
        sensitivity(i)=evaluate(ttest',predict,'sensitivity');
        specificity(i)=evaluate(ttest',predict,'specificity');
        
        
        
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