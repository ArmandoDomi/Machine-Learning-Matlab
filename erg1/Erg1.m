fid = fopen('iris.data', 'r');
a = textscan(fid,'%f %f %f %f %s', 'Delimiter',','); 
fclose(fid);                                             
NumberOfAttributes=length(a);                         
NumberOfPatterns=length(a{1});
t=zeros(1,NumberOfPatterns);



p=[a{1},a{2},a{3},a{4},ones(150,1)]; 
p=p';

for i=1:NumberOfPatterns
    if strcmp('Iris-setosa',char(a{5}(i))) == 1;
        t(i)=1;
    elseif strcmp('Iris-versicolor',char(a{5}(i))) == 1;    
        t(i)=2;
    else
        t(i)=3;
    end
end

figure(1);
hold on;
plot(p(1,1:50),p(3,1:50),'go');
plot(p(1,51:100),p(3,51:100),'y+');
plot(p(1,101:150),p(3,101:150),'m.');
hold off;
indices=crossvalind('Kfold',NumberOfPatterns,9);
figure(2);
hold on;



for i=1:9
    testidx=find(indices==i);
    trainidx=find(indices~=i);
    fprintf('**%d',i);
    fprintf('o Kfold**\n');
    fprintf('Protipa poy anikoun sto test set:\n');
    fprintf('%d\n',length(testidx));
    fprintf('Protipa pou anikoun sto train set:\n');
    fprintf('%d\n',length(trainidx));
    ptrain=p(:,trainidx);      
    ttrain=t(trainidx);        
    ptest=p(:,testidx);        
    ttest=t(testidx);           
    subplot(3,3,i);
    plot(ptrain(1,:),ptrain(3,:),'b+', ...     
         ptest(1,:),ptest(3,:),'ro');  
end

hold off;
figure(3);




for i=1:9
    [trainidx,testidx]=crossvalind('LeaveMOut',NumberOfPatterns,15);
    fprintf('**%d',i);
    fprintf('o LeaveMOut**\n');
    fprintf('Protipa poy anikoun sto test set:\n');
    fprintf('%d\n',length(find(testidx==1)));
    fprintf('Protipa pou anikoun sto train set:\n');
    fprintf('%d\n',length(find(trainidx==1)));
    ptrain=p(:,trainidx);
    ttrain=t(trainidx);
    ptest=p(:,testidx);
    ttest=t(testidx);
    subplot(3,3,i);
    plot(ptrain(1,:),ptrain(3,:),'b+', ...
         ptest(1,:),ptest(3,:),'ro');
    
end    


    

