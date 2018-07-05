function model = nbtrain( x ,t )
    
    x0 = x(t == 1,:);
    x1 = x(t == 0,:);
    t0 = length(find(t == 0));
    t1 = length(find(t == 1));
    
    
    prior_prob(1,1) = t0/length(t);
    prior_prob(2,1) = t1/length(t);
    mesi_timi = zeros(2,4);
    diaspora = zeros(2,4);
    
   for i=1:length(x(1,:))
       
        mesi_timi(1,i) = mean(x0(:,i));
        diaspora(1,i) = std(x0(:,i));
        mesi_timi(2,i) = mean(x1(:,i));
        diaspora(2,i) = std(x1(:,i));
   end
   
   model = struct('prior',prior_prob,'mu',mesi_timi,'sigma',diaspora);

   
end    
    
    
    
    
    
    
    

