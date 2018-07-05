function predict = nbpredict(x, model)
    
    predict = zeros(length(x),1);

    for p=1:length(x)
        L = model.prior(2)/model.prior(1);
        for i=1:length(x(1,:))
            L = L * (normpdf(x(p,i), model.mu(2,i), model.sigma(2,i))/normpdf(x(p,i), model.mu(1,i), model.sigma(1,i)));
        end
        if(L<1)
            predict(p,1) = 1;
        end
    end


end