function w=perceptron( x, t, MAXEPOCHS, beta )

    w=rand(1,length(x(:,1)));
    check=1;
    epoxi=1;

   
   while check==1  && epoxi <= MAXEPOCHS
        check=0;
        for p=1:length(x)
         
            u=dot(w,x(:,p));
            if u >= 0
                y=1;
            else
                y=-1;
            end    
            if t(p) ~= y
             
                w=w+beta*(t(p)-y)*x(:,p)';
                check=1;
            end
            
        end
        epoxi=epoxi+1;
   end   
     
  fprintf('To plithos twn epoxwn einai : %d \n',epoxi);
   

    
