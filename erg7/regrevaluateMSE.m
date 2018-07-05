function value = regrevaluateMSE(t , predict)
    
    value=mean((t-predict).^2);

end