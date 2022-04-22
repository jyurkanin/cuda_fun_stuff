
while true

kr = 2;
ks = 1 + (2*kr);
%V = rand(5,5);
V = (rand(1,1));
W = (-rand(1,1));

K = [0 W W W W; 
     -W 0 V V W; 
     -W -V 0 V W; 
     -W -V -V 0 W; 
     -W -W -W -W 0];
%K = V*K/V;

gs = 8;
S = zeros(gs,gs);
A = zeros(gs*gs,gs*gs);
%A = sym(A);

for x = [1:gs]
    for y = [1:gs]
        grid_idx = ((y-1)*gs)+x;
        for xk = [1:ks]
            for yk = [1:ks]
                temp_x = mod(x + (xk - kr -1) - 1, gs) + 1;
                temp_y = mod(y + (yk - kr -1) - 1, gs);
                temp_grid_idx = (temp_y*gs) + temp_x;
                
                A(grid_idx, temp_grid_idx) = K(yk,xk);
            end
        end
    end
end

%A;
%K
%eig(A)
%eig(K)

if(all(eig(A) <= 0))
    eig(A)
    K
    break;
end

end

%% Same thing no loop.

syms k1 k2 k3 real

kr = 1;
ks = 1 + (2*kr);

%K = sym('K', [ks ks]);
%k1 = 1;
%k2 = .35;
K = [0   k1 k2;
    -k1  0  k3; 
    -k2 -k3 0]


gs = 4;
S = zeros(gs,gs);
A = zeros(gs*gs,gs*gs);
A = sym(A);

for x = [1:gs]
    for y = [1:gs]
        grid_idx = ((y-1)*gs)+x;
        for xk = [1:ks]
            for yk = [1:ks]
                temp_x = mod(x + (xk - kr -1) - 1, gs) + 1;
                temp_y = mod(y + (yk - kr -1) - 1, gs);
                temp_grid_idx = (temp_y*gs) + temp_x;
                
                A(grid_idx, temp_grid_idx) = K(yk,xk);
            end
        end
    end
end

A
K
%eig(A)
%eig(K)






