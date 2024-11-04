Z=-X;
it=100;
w=(1/L)*ones(it,L);
I=eye(L,L);
E=zeros(it,1);
for i=1:(it-1)
    v=(w(i,:)*X')';
    e=y-v;
    w(i+1,:)=w(i,:)-((Z'*Z+n*I)\Z'*e)';
    E(i)=(1/length(e))*(e')*e;
    if(E(i)<0.0001)
        break;
    end
end